use diffusion_rs::{
    api::ModelCtx, model_config::ModelConfig, txt2img_config::Txt2ImgConfig, utils::DiffusionError,
};
use image::RgbImage;
use std::sync::{mpsc::Sender, Arc, Mutex};

pub enum DiffusionOutput {
    Step(String),
    FatalErr(WorkerError),
    Done(Vec<RgbImage>),
}

pub type Model = Arc<Mutex<ModelCtx>>;

pub enum ModelOutput {
    Step(String),
    FatalErr(WorkerError),
    Done(Model),
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Could not determine number of threads available: {0}")]
    ThreadCountError(#[from] std::io::Error),

    #[error("DiffusionError: {0}")]
    DiffusionError(#[from] DiffusionError),

    #[error("Could not send newly generated step preview image out to the game engine.")]
    SendError,
}

pub fn build_model(
    model_config: &ModelConfig,
    model_sender: &Sender<ModelOutput>,
) -> Result<(), WorkerError> {
    model_sender
        .send(ModelOutput::Done(Arc::new(Mutex::new(ModelCtx::new(
            &model_config,
        )?))))
        .map_err(|_| WorkerError::SendError)?;
    Ok(())
}

pub fn start_diffusion_worker(
    model: &Model,
    txt2img_config: &mut Txt2ImgConfig,
    diffusion_sender: &Sender<DiffusionOutput>,
) -> Result<(), WorkerError> {
    let model_ctx = model.lock().unwrap();

    diffusion_sender
        .send(DiffusionOutput::Done(model_ctx.txt2img(txt2img_config)?))
        .map_err(|_| WorkerError::SendError)?;
    Ok(())
}
