use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use image::ImageFormat;
use log::{debug, info};
use rayon::prelude::*;
use tempfile::TempDir;

use crate::subtitles::SubtitleRenderer;

#[cfg(feature = "native-ffmpeg")]
pub fn init_native_ffmpeg() -> Result<()> {
    ffmpeg_next::init().context("failed to initialize ffmpeg-next native bindings")
}

#[cfg(not(feature = "native-ffmpeg"))]
pub fn init_native_ffmpeg() -> Result<()> {
    Ok(())
}

#[derive(Debug, Clone)]
pub struct VideoJob {
    pub input_video: PathBuf,
    pub audio: PathBuf,
    pub output: PathBuf,
    pub fps: Option<f64>,
    pub crf: u8,
    pub audio_bitrate: String,
}

impl VideoJob {
    pub fn validate(&self) -> Result<()> {
        assert_existing_file(&self.input_video, "input video")?;
        assert_existing_file(&self.audio, "audio")?;
        if let Some(parent) = self.output.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
        if !(0..=51).contains(&self.crf) {
            bail!("crf must be between 0 and 51, got {}", self.crf);
        }
        if self.audio_bitrate.trim().is_empty() {
            bail!("audio_bitrate must not be empty");
        }
        Ok(())
    }
}

pub fn render_subtitled_video(job: &VideoJob, renderer: &SubtitleRenderer) -> Result<()> {
    job.validate()?;
    init_native_ffmpeg()?;
    ensure_ffmpeg_tools()?;

    let fps = match job.fps {
        Some(value) if value > 0.0 && value.is_finite() => value,
        _ => probe_fps(&job.input_video).unwrap_or_else(|err| {
            info!("Could not probe FPS from video; falling back to renderer FPS. Error: {err}");
            renderer.fps()
        }),
    };
    if fps <= 0.0 || !fps.is_finite() {
        bail!("resolved fps must be positive and finite, got {fps}");
    }

    let workspace = TempDir::new().context("failed to create temporary frame workspace")?;
    let decoded_dir = workspace.path().join("decoded");
    let rendered_dir = workspace.path().join("rendered");
    fs::create_dir_all(&decoded_dir)?;
    fs::create_dir_all(&rendered_dir)?;

    decode_frames(&job.input_video, &decoded_dir)?;
    let frames = list_png_frames(&decoded_dir)?;
    if frames.is_empty() {
        bail!(
            "ffmpeg decoded zero frames from {}",
            job.input_video.display()
        );
    }
    info!("Decoded {} frames", frames.len());

    render_frames_parallel(&frames, &rendered_dir, renderer)?;

    let silent_video = workspace.path().join("video_with_subtitles.mp4");
    encode_frames(&rendered_dir, fps, job.crf, &silent_video)?;
    mux_audio(&silent_video, &job.audio, &job.output, &job.audio_bitrate)?;
    info!("Final video written to {}", job.output.display());
    Ok(())
}

fn ensure_ffmpeg_tools() -> Result<()> {
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg.arg("-version");
    run_command(&mut ffmpeg, "ffmpeg -version").map(|_| ())?;

    let mut ffprobe = Command::new("ffprobe");
    ffprobe.arg("-version");
    run_command(&mut ffprobe, "ffprobe -version").map(|_| ())?;
    Ok(())
}

fn decode_frames(input_video: &Path, decoded_dir: &Path) -> Result<()> {
    let frame_pattern = decoded_dir.join("frame_%08d.png");
    let mut command = Command::new("ffmpeg");
    command
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(input_video)
        .arg("-vsync")
        .arg("0")
        .arg(frame_pattern);
    run_command(&mut command, "decode video frames").map(|_| ())
}

fn render_frames_parallel(
    frames: &[PathBuf],
    rendered_dir: &Path,
    renderer: &SubtitleRenderer,
) -> Result<()> {
    frames
        .par_iter()
        .enumerate()
        .try_for_each(|(index, frame_path)| -> Result<()> {
            let mut image = image::open(frame_path)
                .with_context(|| format!("failed to open decoded frame {}", frame_path.display()))?
                .to_rgba8();
            renderer.render_frame(&mut image, index as u64);
            let output_path = rendered_dir.join(format!("frame_{:08}.png", index + 1));
            image
                .save_with_format(&output_path, ImageFormat::Png)
                .with_context(|| {
                    format!("failed to save rendered frame {}", output_path.display())
                })?;
            Ok(())
        })
}

fn encode_frames(rendered_dir: &Path, fps: f64, crf: u8, output: &Path) -> Result<()> {
    let frame_pattern = rendered_dir.join("frame_%08d.png");
    let mut command = Command::new("ffmpeg");
    command
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-framerate")
        .arg(format_fps(fps))
        .arg("-i")
        .arg(frame_pattern)
        .arg("-c:v")
        .arg("libx264")
        .arg("-preset")
        .arg("medium")
        .arg("-crf")
        .arg(crf.to_string())
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(output);
    run_command(&mut command, "encode subtitled video").map(|_| ())
}

fn mux_audio(video: &Path, audio: &Path, output: &Path, audio_bitrate: &str) -> Result<()> {
    let mut command = Command::new("ffmpeg");
    command
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(video)
        .arg("-i")
        .arg(audio)
        .arg("-map")
        .arg("0:v:0")
        .arg("-map")
        .arg("1:a:0")
        .arg("-c:v")
        .arg("copy")
        .arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg(audio_bitrate)
        .arg("-shortest")
        .arg("-movflags")
        .arg("+faststart")
        .arg(output);
    run_command(&mut command, "mux audio and video").map(|_| ())
}

fn probe_fps(input_video: &Path) -> Result<f64> {
    let mut command = Command::new("ffprobe");
    command
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=r_frame_rate")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(input_video);
    let stdout = run_command(&mut command, "probe video fps")?;
    parse_fps(stdout.trim())
        .ok_or_else(|| anyhow!("could not parse fps value from ffprobe output: {stdout:?}"))
}

fn parse_fps(raw: &str) -> Option<f64> {
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    if let Some((num, den)) = value.split_once('/') {
        let numerator: f64 = num.parse().ok()?;
        let denominator: f64 = den.parse().ok()?;
        if denominator == 0.0 {
            return None;
        }
        return Some(numerator / denominator);
    }
    value.parse::<f64>().ok()
}

fn list_png_frames(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut frames: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("failed to read frame directory {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| extension.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();
    frames.sort();
    Ok(frames)
}

fn assert_existing_file(path: &Path, label: &str) -> Result<()> {
    if !path.exists() {
        bail!("{label} does not exist: {}", path.display());
    }
    if !path.is_file() {
        bail!("{label} is not a file: {}", path.display());
    }
    if path.metadata()?.len() == 0 {
        bail!("{label} is empty: {}", path.display());
    }
    Ok(())
}

fn run_command(command: &mut Command, label: &str) -> Result<String> {
    debug!("Running {label}: {:?}", command);
    let output = command
        .output()
        .with_context(|| format!("failed to execute {label}"))?;

    if !output.status.success() {
        bail!(
            "{label} failed with status {}\nSTDOUT:\n{}\nSTDERR:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn format_fps(fps: f64) -> String {
    if fps.fract().abs() < f64::EPSILON {
        format!("{}", fps as u64)
    } else {
        format!("{fps:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fractional_fps() {
        let fps = parse_fps("30000/1001").unwrap();
        assert!((fps - 29.970).abs() < 0.01);
    }

    #[test]
    fn parses_decimal_fps() {
        assert_eq!(parse_fps("30").unwrap(), 30.0);
    }
}
