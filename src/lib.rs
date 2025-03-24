use diffusion::ModelOutput;
use diffusion_rs::{
    api::ModelCtx, model_config::ModelConfigBuilder, txt2img_config::Txt2ImgConfigBuilder,
    utils::SdLogLevel,
};
use godot::{
    classes::{Image, ProjectSettings},
    prelude::*,
};
use std::{
    ffi::c_void,
    path::PathBuf,
    sync::{mpsc::Receiver, Arc, Mutex},
};
use utils::{
    godot_log_callback, GRngFunction, GSampleMethod, GSchedule, GWeightType, IntoGImage, TryIntoRGB,
};
mod diffusion;
mod utils;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GodotDiffusionError {
    #[error("Diffusion Error: {0}")]
    DiffusionError(#[from] diffusion_rs::utils::DiffusionError),
    #[error("Model Context Lock was Poisoned")]
    PoisonedLockError,
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
                Ok(diffusion::ModelOutput::Step(_)) => {
                    self.base_mut()
                        .emit_signal("model_step", &[Variant::from("step")]);
                }
                Ok(diffusion::ModelOutput::Done(response)) => {
                    self.model = Some(response);
                    self.base_mut().emit_signal("model_build_complete", &[]);
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
            .log_callback(godot_log_callback as extern "C" fn(SdLogLevel, *const i8, *mut c_void))
            .build()
            .unwrap();

        godot_print!("Config Built");

        let result = || -> Result<(), String> {
            // make and store channels for communicating with the llm worker thread
            let (sender, receiver) = std::sync::mpsc::channel();
            self.model_receiver = Some(receiver);
            // start the llm worker
            std::thread::spawn(move || diffusion::build_model(&model_config, &sender));
            Ok(())
        };

        // run it and show error in godot if it fails
        if let Err(msg) = result() {
            godot_error!("Error running model: {}", msg);
        }
    }

    fn get_model(&self) -> Result<Arc<Mutex<ModelCtx>>, GodotDiffusionError> {
        self.model
            .clone()
            .ok_or(GodotDiffusionError::PoisonedLockError)
    }

    #[signal]
    fn model_step(step: String);

    #[signal]
    fn model_build_complete();
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
struct DiffusionImageGen {
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
    base: Base<Node>,
}

#[godot_api]
impl INode for DiffusionImageGen {
    fn init(base: Base<Node>) -> Self {
        DiffusionImageGen {
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
            base,
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        while let Some(rx) = self.diffusion_receiver.as_ref() {
            match rx.try_recv() {
                Ok(diffusion::DiffusionOutput::Step(step)) => {
                    self.base_mut()
                        .emit_signal("generation_step", &[Variant::from(step)]);
                }
                Ok(diffusion::DiffusionOutput::Done(mut response)) => {
                    let images = response
                        .iter_mut()
                        .map(|f| f.into_g_image())
                        .collect::<Vec<_>>();

                    // convert to godot array
                    let mut images_array = Array::new();
                    for image in images.iter() {
                        images_array.push(image);
                    }

                    self.base_mut()
                        .emit_signal("generation_complete", &[Variant::from(images_array)]);
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
                    // this prevents repeating the dead channel error message foreve
                    self.diffusion_receiver = None;
                }
            }
        }
    }
}

#[godot_api]
impl DiffusionImageGen {
    #[func]
    fn start_txt2img(&mut self) {
        godot_print!("Txt2Img started");

        let control_image = self
            .get_control_condition_image()
            .unwrap()
            .try_into_rgb()
            .unwrap();

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
            .control_cond(control_image)
            .control_strength(self.control_strength)
            .clip_skip(self.clip_skip)
            .batch_count(self.batch_count)
            .seed(self.seed);

        for lora in self.lora_models.iter_shared() {
            let lora_model: &LoraModel = &lora.bind();

            binding.add_lora_model(
                &lora_model.get_lora_model_name().to_string(),
                lora_model.weight,
            );
        }

        let mut txt2img_config = binding.build().unwrap();

        let result = || -> Result<(), String> {
            let (sender, receiver) = std::sync::mpsc::channel();
            self.diffusion_receiver = Some(receiver);

            // Extract the model safely outside the thread closure
            let model_clone = self
                .model_node
                .as_ref()
                .unwrap()
                .bind()
                .get_model()
                .unwrap();
            //spawn a thread to run the model
            std::thread::spawn(move || {
                diffusion::start_diffusion_worker(&model_clone, &mut txt2img_config, &sender)
            });
            Ok(())
        };
        // run it and show error in godot if it fails
        if let Err(msg) = result() {
            godot_error!("Error running model: {}", msg);
        }
    }

    #[signal]
    fn generation_step(step: String);

    #[signal]
    fn generation_complete(images: Array<Gd<Image>>);
}
