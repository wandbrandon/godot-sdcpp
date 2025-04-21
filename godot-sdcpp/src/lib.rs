use diffusion::ModelOutput;
use diffusion_rs::{
    api::ModelCtx, model_config::ModelConfigBuilder, txt2img_config::Txt2ImgConfigBuilder,
};
use godot::{
    classes::{Image, ProjectSettings},
    prelude::*,
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, Sender},
    },
};
use utils::{GRngFunction, GSampleMethod, GSchedule, GWeightType, IntoGImage, TryIntoRGB};
mod diffusion;
mod utils;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GodotDiffusionError {
    #[error("Diffusion Error: {0}")]
    DiffusionError(#[from] diffusion_rs::types::DiffusionError),
}

struct GodotSDCPPExtension;

#[gdextension]
unsafe impl ExtensionLibrary for GodotSDCPPExtension {}

#[derive(GodotClass)]
#[class(init, base=Node)]
/// The model node is used to load the model
struct DiffusionModelContext {
    #[export(file = "*.safetensors")]
    model_path: GString,

    #[export(file = "*.safetensors")]
    clip_l_path: GString,

    #[export(file = "*.safetensors")]
    clip_g_path: GString,

    #[export(file = "*.safetensors")]
    t5xxl_path: GString,

    #[export(file = "*.safetensors")]
    unet_model_path: GString,

    #[export(file = "*.safetensors")]
    vae_path: GString,

    #[export(file = "*.safetensors")]
    taesd_path: GString,

    #[export(file = "*.safetensors")]
    control_net_model_path: GString,

    #[export(dir)]
    lora_model_dir: GString,

    #[export(dir)]
    embeddings_dir: GString,

    #[export(dir)]
    stacked_id_embd_dir: GString,

    #[export]
    vae_decode_only: bool,

    #[export]
    vae_tiling: bool,

    #[export]
    n_threads: i32,

    #[export]
    weight_type: GWeightType,

    #[export]
    rng_type: GRngFunction,

    #[export]
    schedule: GSchedule,

    #[export]
    keep_clip_on_cpu: bool,

    #[export]
    keep_control_net_cpu: bool,

    #[export]
    keep_vae_on_cpu: bool,

    #[export]
    flash_attention: bool,

    model: Option<Arc<Mutex<ModelCtx>>>,

    model_receiver: Option<Receiver<ModelOutput>>,

    base: Base<Node>,
}

#[godot_api]
impl INode for DiffusionModelContext {
    fn physics_process(&mut self, _delta: f64) {
        while let Some(rx) = self.model_receiver.as_ref() {
            match rx.try_recv() {
                Ok(diffusion::ModelOutput::Step(step, steps, time)) => {
                    self.signals().step().emit(step, steps, time);
                }
                Ok(diffusion::ModelOutput::Done(response)) => {
                    self.model = Some(response);
                    self.signals().build_complete().emit();
                }
                Ok(diffusion::ModelOutput::FatalErr(msg)) => {
                    godot_error!("Model worker crashed: {msg}");
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    godot_error!(
                        "Model output channel died. Did the Diffusion Model Build worker crash?"
                    );
                    self.model_receiver = None;
                }
            }
        }
    }
}

#[godot_api]
impl DiffusionModelContext {
    #[func]
    fn start_model_build(&mut self) {
        let project_settings = ProjectSettings::singleton();
        let model = PathBuf::from(
            project_settings
                .globalize_path(&self.model_path)
                .to_string(),
        );
        let clip_l = PathBuf::from(
            project_settings
                .globalize_path(&self.clip_l_path)
                .to_string(),
        );
        let clip_g = PathBuf::from(
            project_settings
                .globalize_path(&self.clip_g_path)
                .to_string(),
        );
        let t5xxl = PathBuf::from(
            project_settings
                .globalize_path(&self.t5xxl_path)
                .to_string(),
        );
        let diffusion_model = PathBuf::from(
            project_settings
                .globalize_path(&self.unet_model_path)
                .to_string(),
        );
        let vae = PathBuf::from(project_settings.globalize_path(&self.vae_path).to_string());
        let taesd = PathBuf::from(
            project_settings
                .globalize_path(&self.taesd_path)
                .to_string(),
        );
        let control_net = PathBuf::from(
            project_settings
                .globalize_path(&self.control_net_model_path)
                .to_string(),
        );
        let lora_model_dir = PathBuf::from(
            project_settings
                .globalize_path(&self.lora_model_dir)
                .to_string(),
        );
        let embeddings_dir = PathBuf::from(
            project_settings
                .globalize_path(&self.embeddings_dir)
                .to_string(),
        );
        let stacked_id_embd_dir = PathBuf::from(
            project_settings
                .globalize_path(&self.stacked_id_embd_dir)
                .to_string(),
        );

        godot_print!("Building Config");

        // make and store channels for communicating with the llm worker thread
        let (sender, receiver) = std::sync::mpsc::channel();
        self.model_receiver = Some(receiver);
        // Clone the sender for use in the model worker
        let sender_clone = sender.clone();

        let strtest = Arc::new(Mutex::new(String::new()));

        let model_config = ModelConfigBuilder::default()
            .model(model)
            .clip_l(clip_l)
            .clip_g(clip_g)
            .t5xxl(t5xxl)
            .diffusion_model(diffusion_model)
            .vae(vae)
            .taesd(taesd)
            .control_net(control_net)
            .lora_model_dir(lora_model_dir)
            .embeddings_dir(embeddings_dir)
            .stacked_id_embd_dir(stacked_id_embd_dir)
            .vae_decode_only(self.vae_decode_only)
            .n_threads(self.n_threads)
            .weight_type(self.weight_type)
            .rng_type(self.rng_type)
            .schedule(self.schedule)
            .keep_clip_on_cpu(self.keep_clip_on_cpu)
            .keep_control_net_cpu(self.keep_control_net_cpu)
            .keep_vae_on_cpu(self.keep_vae_on_cpu)
            .flash_attention(self.flash_attention)
            .build()
            .unwrap();

        ModelCtx::set_log_callback(|level, text| {
            let trim_text = text.trim();
            match level {
                diffusion_rs::types::SdLogLevel::SD_LOG_DEBUG => godot_print!("{trim_text}"),
                diffusion_rs::types::SdLogLevel::SD_LOG_INFO => godot_print!("{trim_text}"),
                diffusion_rs::types::SdLogLevel::SD_LOG_WARN => godot_warn!("{trim_text}"),
                diffusion_rs::types::SdLogLevel::SD_LOG_ERROR => godot_error!("{trim_text}"),
                _ => godot_print!("{trim_text}"),
            }
        });

        ModelCtx::set_progress_callback(move |step, steps, time| {
            sender_clone
                .send(ModelOutput::Step(step, steps, time))
                .expect("Failed to send progress message")
        });

        godot_print!("{strtest:?}");

        godot_print!("Starting Model Worker");
        std::thread::spawn(move || diffusion::run_model_worker(&model_config, &sender));
    }

    #[signal]
    fn step(step: i32, steps: i32, time: f32);

    #[signal]
    fn build_complete();
}

#[derive(GodotClass)]
#[class(tool, init, base=Resource)]
struct LoraModel {
    #[export]
    lora_model_name: GString,
    #[export]
    weight: f32,
    base: Base<Resource>,
}

#[derive(GodotClass)]
#[class(base=Node)]
struct DiffusionImageGenerator {
    #[export]
    /// The model node for the txt2img.
    model_node: Option<Gd<DiffusionModelContext>>,

    /// The prompt for the txt2img.
    #[export]
    #[var(hint = MULTILINE_TEXT)]
    prompt: GString,

    /// The negative prompt (default: "")
    #[export]
    #[var(hint = MULTILINE_TEXT)]
    pub negative_prompt: GString,

    /// Suffix that needs to be added to prompt (e.g. lora model)
    #[export]
    lora_models: Array<Gd<LoraModel>>,

    /// Ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)
    /// <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
    #[export]
    pub clip_skip: i32,

    /// Unconditional guidance scale (default: 7.0)
    #[export]
    pub cfg_scale: f32,

    /// Guidance (default: 3.5) for Flux/DiT models
    #[export]
    pub guidance: f32,

    /// eta in DDIM, only for DDIM and TCD: (default: 0)
    #[export]
    pub eta: f32,

    /// Image height, in pixel space (default: 512)
    #[export]
    pub height: i32,

    /// Image width, in pixel space (default: 512)
    #[export]
    pub width: i32,

    /// Sampling-method (default: EULER_A)
    #[export]
    pub sample_method: GSampleMethod,

    /// Number of sample steps (default: 20)
    #[export]
    pub sample_steps: i32,

    /// RNG seed (default: 42, use random seed for < 0)
    #[export]
    pub seed: i64,

    /// Number of images to generate (default: 1)
    #[export]
    pub batch_count: i32,

    /// Control condition image (default: None)
    #[export]
    pub control_condition_image: Option<Gd<Image>>,

    /// Strength to apply Control Net (default: 0.9)
    /// 1.0 corresponds to full destruction of information in init
    #[export]
    pub control_strength: f32,

    /// Strength for keeping input identity (default: 20%)
    #[export]
    pub style_strength: f32,

    // /// Normalize PHOTOMAKER input id images
    // #[export]
    // pub normalize_input: bool,

    // /// Path to PHOTOMAKER input id images dir
    // #[export(dir)]
    // pub input_id_images: GString,
    /// Layers to skip for SLG steps: (default: [7,8,9])
    #[export]
    pub skip_layer: PackedInt32Array,

    /// skip layer guidance (SLG) scale, only for DiT models: (default: 0)
    /// 0 means disabled, a value of 2.5 is nice for sd3.5 medium
    #[export]
    pub slg_scale: f32,

    /// SLG enabling point: (default: 0.01)
    #[export]
    pub skip_layer_start: f32,

    /// SLG disabling point: (default: 0.2)
    #[export]
    pub skip_layer_end: f32,

    diffusion_receiver: Option<Receiver<diffusion::DiffusionOutput>>,
    diffusion_sender: Option<Sender<diffusion::DiffusionOutput>>,
    base: Base<Node>,
}

#[godot_api]
impl INode for DiffusionImageGenerator {
    fn init(base: Base<Node>) -> Self {
        DiffusionImageGenerator {
            model_node: None,
            prompt: "".into(),
            lora_models: Array::new(),
            negative_prompt: "".into(),
            clip_skip: 0,
            cfg_scale: 7.0,
            guidance: 3.5,
            eta: 1.0,
            height: 512,
            width: 512,
            sample_method: GSampleMethod::EulerA,
            sample_steps: 20,
            seed: 42,
            batch_count: 1,
            control_condition_image: None,
            control_strength: 0.9,
            style_strength: 0.2,
            skip_layer: PackedInt32Array::new(),
            slg_scale: 0.0,
            skip_layer_start: 0.01,
            skip_layer_end: 0.2,
            diffusion_receiver: None,
            diffusion_sender: None,
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        while let Some(rx) = self.diffusion_receiver.as_ref() {
            match rx.try_recv() {
                Ok(diffusion::DiffusionOutput::Done(mut response)) => {
                    let images = response
                        .iter_mut()
                        .map(|f| f.into_g_image())
                        .collect::<Array<_>>();

                    self.signals().complete().emit(images);
                }
                Ok(diffusion::DiffusionOutput::FatalErr(msg)) => {
                    godot_error!("Diffusion worker crashed: {msg}");
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    godot_error!("Diffusion output channel died. Did the Diffusion worker crash?");
                    // set hanging channel to None
                    // this prevents repeating the dead channel error message forever
                    self.diffusion_receiver = None;
                }
            }
        }
    }
}

#[godot_api]
impl DiffusionImageGenerator {
    #[func]
    fn start_txt2img_worker(&mut self) {
        let mut binding = Txt2ImgConfigBuilder::default();
        binding
            .prompt(self.prompt.to_string())
            .negative_prompt(self.negative_prompt.to_string())
            .height(self.height)
            .width(self.width)
            .cfg_scale(self.cfg_scale)
            .eta(self.eta)
            .sample_method(self.sample_method)
            .sample_steps(self.sample_steps)
            .control_strength(self.control_strength)
            .clip_skip(self.clip_skip)
            .batch_count(self.batch_count)
            .seed(self.seed);

        if let Some(mut control_condition_image) = self.get_control_condition_image() {
            let control_cond = control_condition_image
                .try_into_rgb()
                .expect("Error: Image is not RGB!");

            binding.control_cond(control_cond);
        }

        for lora in self.lora_models.iter_shared() {
            let lora_model: &LoraModel = &lora.bind();

            binding.add_lora_model(
                &lora_model.get_lora_model_name().to_string(),
                lora_model.weight,
            );
        }

        let text2image_config = binding.build().expect("Failed to build config");

        let (sender, receiver) = std::sync::mpsc::channel();
        self.diffusion_receiver = Some(receiver);
        self.diffusion_sender = Some(sender.clone());

        // Extract the model safely outside the thread closure
        let model_clone = self
            .model_node
            .as_ref()
            .expect("Couldn't get reference to model node")
            .bind()
            .model
            .clone()
            .expect("Model Context is not initialized!");

        //spawn a thread to run the model
        std::thread::spawn(move || {
            diffusion::run_diffusion_worker(&model_clone, &text2image_config, &sender)
        });
    }

    #[signal]
    fn complete(images: Array<Gd<Image>>);
}
