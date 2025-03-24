// use diffusion_rs::api::{txt2img, ClipSkip, ConfigBuilder, SampleMethod, Schedule, WeightType};
// use std::path::Path;

// fn main() {
//     let config = ConfigBuilder::default()
//         //.flash_attenuation(true)
//         // get path to the entire model
//         .model(Path::new("models/mistoonAnime_v30.safetensors"))
//         //get path of lora model
//         .lora_model(Path::new(
//             "models/loras/pcm_sd15_lcmlike_lora_converted.safetensors",
//         ))
//         //get path of taesd model
//         .taesd(Path::new("models/taesd1.safetensors"))
//         .steps(2)
//         .strength(1.0)
//         .prompt(
//             "masterpiece, best quality, absurdres, 1girl, succubus, bobcut, black hair, horns, portrait, purple skin, black eyes",
//         )
//         .clip_skip(ClipSkip::OneLayer)
//         .height(384)
//         .width(384)
//         .schedule(Schedule::AYS)
//         .sampling_method(SampleMethod::LCM)
//         .cfg_scale(1.0)
//         .weight_type(WeightType::SD_TYPE_Q8_0)
//         .build()
//         .expect("Error when Building Config");

//     txt2img(config).expect("Error when generating image");
// }
