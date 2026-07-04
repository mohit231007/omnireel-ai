mod subtitles;
mod video;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use log::info;
use subtitles::{SubtitleRenderer, SubtitleStyle};
use video::{render_subtitled_video, VideoJob};

#[derive(Debug, Parser)]
#[command(name = "omnireel-ai-engine")]
#[command(about = "Rust video compositor for OmniReel AI", long_about = None)]
struct Cli {
    #[arg(long)]
    input_video: PathBuf,
    #[arg(long)]
    audio: PathBuf,
    #[arg(long)]
    subtitles: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    font: Option<PathBuf>,
    #[arg(long)]
    fps: Option<f64>,
    #[arg(long, default_value_t = 18)]
    crf: u8,
    #[arg(long, default_value = "192k")]
    audio_bitrate: String,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value_t = 42.0)]
    font_scale: f32,
    #[arg(long, default_value_t = 0.86)]
    max_width_ratio: f32,
    #[arg(long, default_value_t = 0.105)]
    margin_bottom_ratio: f32,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();

    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("failed to configure rayon global thread pool")?;
    }

    let font_path = cli.font.clone().unwrap_or_else(default_font_path);
    let fps = cli.fps.unwrap_or(30.0);
    let style = SubtitleStyle {
        font_scale: cli.font_scale,
        max_width_ratio: cli.max_width_ratio,
        margin_bottom_ratio: cli.margin_bottom_ratio,
        ..SubtitleStyle::default()
    };

    info!("Loading subtitles from {}", cli.subtitles.display());
    info!("Using font {}", font_path.display());
    let renderer = SubtitleRenderer::from_whisper_json(&cli.subtitles, &font_path, fps, style)?;

    let job = VideoJob {
        input_video: cli.input_video,
        audio: cli.audio,
        output: cli.output,
        fps: cli.fps,
        crf: cli.crf,
        audio_bitrate: cli.audio_bitrate,
    };

    render_subtitled_video(&job, &renderer)?;
    Ok(())
}

fn default_font_path() -> PathBuf {
    let candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "./assets/fonts/DejaVuSans-Bold.ttf",
    ];
    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    PathBuf::from("./assets/fonts/DejaVuSans-Bold.ttf")
}
