use std::cmp::Ordering;
use std::fs;
use std::path::Path;

use ab_glyph::{FontArc, PxScale};
use anyhow::{anyhow, Context, Result};
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct SubtitleStyle {
    pub font_scale: f32,
    pub margin_bottom_ratio: f32,
    pub max_width_ratio: f32,
    pub line_gap_px: u32,
    pub padding_px: u32,
    pub text_color: Rgba<u8>,
    pub shadow_color: Rgba<u8>,
    pub box_color: Rgba<u8>,
}

impl Default for SubtitleStyle {
    fn default() -> Self {
        Self {
            font_scale: 42.0,
            margin_bottom_ratio: 0.105,
            max_width_ratio: 0.86,
            line_gap_px: 8,
            padding_px: 18,
            text_color: Rgba([255, 255, 255, 255]),
            shadow_color: Rgba([0, 0, 0, 210]),
            box_color: Rgba([0, 0, 0, 165]),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubtitleCue {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub start_frame: u64,
    pub end_frame: u64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct SubtitleRenderer {
    cues: Vec<SubtitleCue>,
    font: FontArc,
    style: SubtitleStyle,
    fps: f64,
}

#[derive(Debug, Deserialize)]
struct WhisperDocument {
    #[serde(default)]
    segments: Vec<WhisperSegment>,
}

#[derive(Debug, Deserialize)]
struct WhisperSegment {
    start: f64,
    end: f64,
    #[serde(default)]
    text: String,
    #[serde(default)]
    words: Vec<WhisperWord>,
}

#[derive(Debug, Deserialize)]
struct WhisperWord {
    start: Option<f64>,
    end: Option<f64>,
    #[serde(alias = "text", alias = "token")]
    word: String,
}

impl SubtitleRenderer {
    pub fn from_whisper_json(path: &Path, font_path: &Path, fps: f64, style: SubtitleStyle) -> Result<Self> {
        if fps <= 0.0 || !fps.is_finite() {
            return Err(anyhow!("fps must be a positive finite number, got {fps}"));
        }

        let font_bytes = fs::read(font_path)
            .with_context(|| format!("failed to read font file {}", font_path.display()))?;
        let font = FontArc::try_from_vec(font_bytes)
            .map_err(|_| anyhow!("failed to parse font file {}", font_path.display()))?;

        let json = fs::read_to_string(path)
            .with_context(|| format!("failed to read subtitle JSON {}", path.display()))?;
        let doc: WhisperDocument = serde_json::from_str(&json)
            .with_context(|| format!("failed to parse Whisper-compatible JSON {}", path.display()))?;
        let cues = cues_from_whisper(doc, fps)?;
        if cues.is_empty() {
            return Err(anyhow!("no subtitle cues were parsed from {}", path.display()));
        }

        Ok(Self { cues, font, style, fps })
    }

    pub fn render_frame(&self, frame: &mut RgbaImage, frame_index: u64) {
        if let Some(cue) = self.cue_for_frame(frame_index) {
            self.draw_cue(frame, cue);
        }
    }

    pub fn cue_for_frame(&self, frame_index: u64) -> Option<&SubtitleCue> {
        let result = self.cues.binary_search_by(|cue| {
            if frame_index < cue.start_frame {
                Ordering::Greater
            } else if frame_index > cue.end_frame {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        });
        result.ok().map(|idx| &self.cues[idx])
    }

    pub fn fps(&self) -> f64 { self.fps }

    fn draw_cue(&self, frame: &mut RgbaImage, cue: &SubtitleCue) {
        let image_width = frame.width();
        let image_height = frame.height();
        if image_width == 0 || image_height == 0 { return; }

        let max_text_width = ((image_width as f32) * self.style.max_width_ratio).round() as u32;
        let scale = PxScale::from(self.style.font_scale);
        let lines = self.wrap_text(&cue.text, max_text_width.max(1));
        if lines.is_empty() { return; }

        let line_heights: Vec<u32> = lines.iter().map(|line| {
            let (_, height) = text_size(scale, &self.font, line);
            height.max(self.style.font_scale.round() as u32)
        }).collect();
        let text_block_height: u32 = line_heights.iter().sum::<u32>()
            + self.style.line_gap_px.saturating_mul(lines.len().saturating_sub(1) as u32);
        let box_height = text_block_height.saturating_add(self.style.padding_px * 2);
        let box_width = max_text_width.saturating_add(self.style.padding_px * 2).min(image_width);

        let bottom_margin = ((image_height as f32) * self.style.margin_bottom_ratio).round() as u32;
        let box_x = image_width.saturating_sub(box_width) / 2;
        let box_y = image_height.saturating_sub(bottom_margin).saturating_sub(box_height);

        let rect = Rect::at(box_x as i32, box_y as i32).of_size(box_width, box_height);
        draw_filled_rect_mut(frame, rect, self.style.box_color);

        let mut cursor_y = box_y.saturating_add(self.style.padding_px);
        for (idx, line) in lines.iter().enumerate() {
            let (line_width, _) = text_size(scale, &self.font, line);
            let line_x = image_width.saturating_sub(line_width) / 2;
            draw_text_mut(frame, self.style.shadow_color, line_x as i32 + 2, cursor_y as i32 + 2, scale, &self.font, line);
            draw_text_mut(frame, self.style.text_color, line_x as i32, cursor_y as i32, scale, &self.font, line);
            cursor_y = cursor_y.saturating_add(line_heights[idx]).saturating_add(self.style.line_gap_px);
        }
    }

    fn wrap_text(&self, text: &str, max_width: u32) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() { return Vec::new(); }

        let scale = PxScale::from(self.style.font_scale);
        let mut lines: Vec<String> = Vec::new();
        let mut current = String::new();

        for word in words {
            let candidate = if current.is_empty() { word.to_owned() } else { format!("{} {}", current, word) };
            let (width, _) = text_size(scale, &self.font, &candidate);
            if width <= max_width || current.is_empty() {
                current = candidate;
            } else {
                lines.push(current);
                current = word.to_owned();
            }
        }

        if !current.is_empty() { lines.push(current); }
        lines
    }
}

fn cues_from_whisper(doc: WhisperDocument, fps: f64) -> Result<Vec<SubtitleCue>> {
    let mut cues: Vec<SubtitleCue> = Vec::new();

    for segment in doc.segments {
        if !segment.words.is_empty() {
            cues.extend(cues_from_words(segment.words, fps));
            continue;
        }

        let text = normalize_subtitle_text(&segment.text);
        if text.is_empty() || segment.end <= segment.start { continue; }
        cues.push(build_cue(segment.start, segment.end, text, fps));
    }

    cues.sort_by_key(|cue| cue.start_frame);
    coalesce_short_cues(&mut cues, fps);
    Ok(cues)
}

fn cues_from_words(words: Vec<WhisperWord>, fps: f64) -> Vec<SubtitleCue> {
    let mut cues = Vec::new();
    let mut current_words: Vec<String> = Vec::new();
    let mut start: Option<f64> = None;
    let mut end: Option<f64> = None;

    for word in words {
        let word_text = normalize_subtitle_text(&word.word);
        if word_text.is_empty() { continue; }
        let Some(word_start) = word.start else { continue };
        let Some(word_end) = word.end else { continue };
        if word_end <= word_start { continue; }

        if start.is_none() { start = Some(word_start); }
        end = Some(word_end);
        current_words.push(word_text);

        let reached_phrase_len = current_words.len() >= 8;
        let sentence_break = current_words.last()
            .map(|token| token.ends_with('.') || token.ends_with('!') || token.ends_with('?'))
            .unwrap_or(false);
        if reached_phrase_len || sentence_break {
            if let (Some(s), Some(e)) = (start, end) {
                cues.push(build_cue(s, e, current_words.join(" "), fps));
            }
            current_words.clear();
            start = None;
            end = None;
        }
    }

    if !current_words.is_empty() {
        if let (Some(s), Some(e)) = (start, end) {
            cues.push(build_cue(s, e, current_words.join(" "), fps));
        }
    }
    cues
}

fn build_cue(start_seconds: f64, end_seconds: f64, text: String, fps: f64) -> SubtitleCue {
    let start_frame = seconds_to_frame_floor(start_seconds, fps);
    let mut end_frame = seconds_to_frame_ceil(end_seconds, fps).saturating_sub(1);
    if end_frame < start_frame { end_frame = start_frame; }
    SubtitleCue { start_seconds, end_seconds, start_frame, end_frame, text }
}

fn seconds_to_frame_floor(seconds: f64, fps: f64) -> u64 {
    (seconds.max(0.0) * fps).floor() as u64
}

fn seconds_to_frame_ceil(seconds: f64, fps: f64) -> u64 {
    (seconds.max(0.0) * fps).ceil() as u64
}

fn normalize_subtitle_text(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ").trim().to_owned()
}

fn coalesce_short_cues(cues: &mut Vec<SubtitleCue>, fps: f64) {
    if cues.len() < 2 { return; }

    let min_frames = (fps * 0.45).round() as u64;
    let mut merged: Vec<SubtitleCue> = Vec::with_capacity(cues.len());

    for cue in cues.drain(..) {
        if let Some(last) = merged.last_mut() {
            let last_duration = last.end_frame.saturating_sub(last.start_frame);
            let small_gap = cue.start_frame.saturating_sub(last.end_frame) <= (fps * 0.16).round() as u64;
            if last_duration < min_frames && small_gap {
                last.end_seconds = cue.end_seconds;
                last.end_frame = cue.end_frame;
                last.text = format!("{} {}", last.text, cue.text);
                continue;
            }
        }
        merged.push(cue);
    }

    *cues = merged;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_conversion_is_precise() {
        assert_eq!(seconds_to_frame_floor(1.0, 30.0), 30);
        assert_eq!(seconds_to_frame_ceil(1.001, 30.0), 31);
    }

    #[test]
    fn normalizes_text() {
        assert_eq!(normalize_subtitle_text(" hello   world \n "), "hello world");
    }
}
