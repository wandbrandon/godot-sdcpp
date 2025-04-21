use diffusion_rs::{
    api::ModelCtx, model_config::ModelConfig, txt2img_config::Txt2ImgConfig, types::DiffusionError,
};
use image::RgbImage;
use std::sync::{Arc, LazyLock, Mutex, mpsc::Sender};

pub type Model = Arc<Mutex<ModelCtx>>;

// Since stable-diffusion.cpp logging callback functions are global, then we will need to lock the models when building, and lock them while inferencing.
// Will remove this once we have a better solution for logging.
static GLOBAL_MODEL_WORKER_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[derive(Debug)]
pub enum ModelOutput {
    Step(i32, i32, f32),
    FatalErr(WorkerError),
    Done(Model),
}

#[derive(Debug)]
pub enum DiffusionOutput {
    //Step(i32, i32, f32),
    FatalErr(WorkerError),
    Done(Vec<RgbImage>),
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Could not determine number of threads available: {0}")]
    ThreadCountError(#[from] std::io::Error),

    #[error("DiffusionError: {0}")]
    DiffusionError(#[from] DiffusionError),

    #[error("Send Error to channel")]
    SendError,

    #[error("Global Model Lock was poisoned.")]
    GMLPoisonError,

    #[error("Regular Model Lock was poisoned.")]
    ModelLockPoisonError,
}

pub fn run_model_worker(model_config: &ModelConfig, model_sender: &Sender<ModelOutput>) {
    if let Err(e) = model_worker(model_config, model_sender) {
        model_sender
            .send(ModelOutput::FatalErr(e))
            .expect("Fatal Error with Model Output Sender");
    }
}

pub fn model_worker(
    model_config: &ModelConfig,
    model_sender: &Sender<ModelOutput>,
) -> Result<(), WorkerError> {
    // Lock the global model worker lock to ensure that we are not building/inferencing.
    let _lock = GLOBAL_MODEL_WORKER_LOCK
        .lock()
        .map_err(|_| WorkerError::GMLPoisonError)?;

    let model_ctx = ModelCtx::new(&model_config)?;
    let model = Arc::new(Mutex::new(model_ctx));

    model_sender
        .send(ModelOutput::Done(model))
        .map_err(|_| WorkerError::SendError)?;

    Ok(())
}

pub fn run_diffusion_worker(
    model: &Model,
    txt2img_config: &Txt2ImgConfig,
    diffusion_sender: &Sender<DiffusionOutput>,
) {
    if let Err(e) = diffusion_worker(model, txt2img_config, diffusion_sender) {
        diffusion_sender
            .send(DiffusionOutput::FatalErr(e))
            .expect("Fatal Error with Diffusion Sender")
    };
}

pub fn diffusion_worker(
    model: &Model,
    txt2img_config: &Txt2ImgConfig,
    diffusion_sender: &Sender<DiffusionOutput>,
) -> Result<(), WorkerError> {
    // Lock the global model worker lock to ensure that we are not building another model while inferencing.
    let _lock = GLOBAL_MODEL_WORKER_LOCK
        .lock()
        .map_err(|_| WorkerError::GMLPoisonError)?;

    let model_ctx = model
        .lock()
        .map_err(|_| WorkerError::ModelLockPoisonError)?;

    let txt2img_output = model_ctx.txt2img(txt2img_config)?;

    diffusion_sender
        .send(DiffusionOutput::Done(txt2img_output))
        .map_err(|_| WorkerError::SendError)?;

    Ok(())
}
