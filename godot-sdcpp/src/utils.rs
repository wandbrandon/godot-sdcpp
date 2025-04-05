use diffusion_rs::types::{RngFunction, SampleMethod, Schedule, WeightType};
use godot::{
    classes::{image::Format, Image},
    global::Error,
    prelude::*,
};

use image::RgbImage;

#[derive(GodotConvert, Debug, Var, Export, Copy, Clone, Default)]
#[godot(via = GString)]
pub enum GWeightType {
    SdTypeF32,
    #[default]
    SdTypeF16,
    SdTypeQ4_0,
    SdTypeQ4_1,
    SdTypeQ5_0,
    SdTypeQ5_1,
    SdTypeQ8_0,
    SdTypeQ8_1,
    SdTypeQ2K,
    SdTypeQ3K,
    SdTypeQ4K,
    SdTypeQ5K,
    SdTypeQ6K,
    SdTypeQ8K,
    SdTypeIq2Xxs,
    SdTypeIq2Xs,
    SdTypeIq3Xxs,
    SdTypeIq1S,
    SdTypeIq4Nl,
    SdTypeIq3S,
    SdTypeIq2S,
    SdTypeIq4Xs,
    SdTypeI8,
    SdTypeI16,
    SdTypeI32,
    SdTypeI64,
    SdTypeF64,
    SdTypeIq1M,
    SdTypeBf16,
    SdTypeTq1_0,
    SdTypeTq2_0,
    SdTypeCount,
}

impl From<WeightType> for GWeightType {
    fn from(weight_type: WeightType) -> Self {
        match weight_type {
            WeightType::SD_TYPE_F32 => GWeightType::SdTypeF32,
            WeightType::SD_TYPE_F16 => GWeightType::SdTypeF16,
            WeightType::SD_TYPE_Q4_0 => GWeightType::SdTypeQ4_0,
            WeightType::SD_TYPE_Q4_1 => GWeightType::SdTypeQ4_1,
            WeightType::SD_TYPE_Q5_0 => GWeightType::SdTypeQ5_0,
            WeightType::SD_TYPE_Q5_1 => GWeightType::SdTypeQ5_1,
            WeightType::SD_TYPE_Q8_0 => GWeightType::SdTypeQ8_0,
            WeightType::SD_TYPE_Q8_1 => GWeightType::SdTypeQ8_1,
            WeightType::SD_TYPE_Q2_K => GWeightType::SdTypeQ2K,
            WeightType::SD_TYPE_Q3_K => GWeightType::SdTypeQ3K,
            WeightType::SD_TYPE_Q4_K => GWeightType::SdTypeQ4K,
            WeightType::SD_TYPE_Q5_K => GWeightType::SdTypeQ5K,
            WeightType::SD_TYPE_Q6_K => GWeightType::SdTypeQ6K,
            WeightType::SD_TYPE_Q8_K => GWeightType::SdTypeQ8K,
            WeightType::SD_TYPE_IQ2_XXS => GWeightType::SdTypeIq2Xxs,
            WeightType::SD_TYPE_IQ2_XS => GWeightType::SdTypeIq2Xs,
            WeightType::SD_TYPE_IQ3_XXS => GWeightType::SdTypeIq3Xxs,
            WeightType::SD_TYPE_IQ1_S => GWeightType::SdTypeIq1S,
            WeightType::SD_TYPE_IQ4_NL => GWeightType::SdTypeIq4Nl,
            WeightType::SD_TYPE_IQ3_S => GWeightType::SdTypeIq3S,
            WeightType::SD_TYPE_IQ4_XS => GWeightType::SdTypeIq4Xs,
            WeightType::SD_TYPE_IQ2_S => GWeightType::SdTypeIq2S,
            WeightType::SD_TYPE_I8 => GWeightType::SdTypeI8,
            WeightType::SD_TYPE_I16 => GWeightType::SdTypeI16,
            WeightType::SD_TYPE_I32 => GWeightType::SdTypeI32,
            WeightType::SD_TYPE_I64 => GWeightType::SdTypeI64,
            WeightType::SD_TYPE_F64 => GWeightType::SdTypeF64,
            WeightType::SD_TYPE_IQ1_M => GWeightType::SdTypeIq1M,
            WeightType::SD_TYPE_BF16 => GWeightType::SdTypeBf16,
            WeightType::SD_TYPE_TQ1_0 => GWeightType::SdTypeTq1_0,
            WeightType::SD_TYPE_TQ2_0 => GWeightType::SdTypeTq2_0,
            WeightType::SD_TYPE_COUNT => GWeightType::SdTypeCount,
            _ => GWeightType::SdTypeCount,
        }
    }
}

impl Into<WeightType> for GWeightType {
    fn into(self) -> WeightType {
        match self {
            GWeightType::SdTypeF32 => WeightType::SD_TYPE_F32,
            GWeightType::SdTypeF16 => WeightType::SD_TYPE_F16,
            GWeightType::SdTypeQ4_0 => WeightType::SD_TYPE_Q4_0,
            GWeightType::SdTypeQ4_1 => WeightType::SD_TYPE_Q4_1,
            GWeightType::SdTypeQ5_0 => WeightType::SD_TYPE_Q5_0,
            GWeightType::SdTypeQ5_1 => WeightType::SD_TYPE_Q5_1,
            GWeightType::SdTypeQ8_0 => WeightType::SD_TYPE_Q8_0,
            GWeightType::SdTypeQ8_1 => WeightType::SD_TYPE_Q8_1,
            GWeightType::SdTypeQ2K => WeightType::SD_TYPE_Q2_K,
            GWeightType::SdTypeQ3K => WeightType::SD_TYPE_Q3_K,
            GWeightType::SdTypeQ4K => WeightType::SD_TYPE_Q4_K,
            GWeightType::SdTypeQ5K => WeightType::SD_TYPE_Q5_K,
            GWeightType::SdTypeQ6K => WeightType::SD_TYPE_Q6_K,
            GWeightType::SdTypeQ8K => WeightType::SD_TYPE_Q8_K,
            GWeightType::SdTypeIq2Xxs => WeightType::SD_TYPE_IQ2_XXS,
            GWeightType::SdTypeIq2Xs => WeightType::SD_TYPE_IQ2_XS,
            GWeightType::SdTypeIq3Xxs => WeightType::SD_TYPE_IQ3_XXS,
            GWeightType::SdTypeIq1S => WeightType::SD_TYPE_IQ1_S,
            GWeightType::SdTypeIq4Nl => WeightType::SD_TYPE_IQ4_NL,
            GWeightType::SdTypeIq3S => WeightType::SD_TYPE_IQ3_S,
            GWeightType::SdTypeIq4Xs => WeightType::SD_TYPE_IQ4_XS,
            GWeightType::SdTypeIq2S => WeightType::SD_TYPE_IQ2_S,
            GWeightType::SdTypeI8 => WeightType::SD_TYPE_I8,
            GWeightType::SdTypeI16 => WeightType::SD_TYPE_I16,
            GWeightType::SdTypeI32 => WeightType::SD_TYPE_I32,
            GWeightType::SdTypeI64 => WeightType::SD_TYPE_I64,
            GWeightType::SdTypeF64 => WeightType::SD_TYPE_F64,
            GWeightType::SdTypeIq1M => WeightType::SD_TYPE_IQ1_M,
            GWeightType::SdTypeBf16 => WeightType::SD_TYPE_BF16,
            GWeightType::SdTypeTq1_0 => WeightType::SD_TYPE_TQ1_0,
            GWeightType::SdTypeTq2_0 => WeightType::SD_TYPE_TQ2_0,
            GWeightType::SdTypeCount => WeightType::SD_TYPE_COUNT,
        }
    }
}

#[repr(u32)]
#[derive(GodotConvert, Debug, Var, Export, Copy, Clone, Default)]
#[godot(via = u32)]
pub enum GRngFunction {
    StdDefaultRng,
    #[default]
    CudaRng,
}

impl From<RngFunction> for GRngFunction {
    fn from(rng_function: RngFunction) -> Self {
        match rng_function {
            RngFunction::STD_DEFAULT_RNG => GRngFunction::StdDefaultRng,
            RngFunction::CUDA_RNG => GRngFunction::CudaRng,
            _ => GRngFunction::CudaRng,
        }
    }
}

impl Into<RngFunction> for GRngFunction {
    fn into(self) -> RngFunction {
        match self {
            GRngFunction::StdDefaultRng => RngFunction::STD_DEFAULT_RNG,
            GRngFunction::CudaRng => RngFunction::CUDA_RNG,
        }
    }
}

#[derive(GodotConvert, Debug, Var, Export, Copy, Clone, Default)]
#[godot(via = GString)]
pub enum GSchedule {
    #[default]
    Default,
    Discrete,
    Karras,
    Exponential,
    AlignYourSteps,
    GITS,
    NSchedules,
}

impl From<Schedule> for GSchedule {
    fn from(schedule: Schedule) -> Self {
        match schedule {
            Schedule::DEFAULT => GSchedule::Default,
            Schedule::DISCRETE => GSchedule::Discrete,
            Schedule::KARRAS => GSchedule::Karras,
            Schedule::EXPONENTIAL => GSchedule::Exponential,
            Schedule::AYS => GSchedule::AlignYourSteps,
            Schedule::GITS => GSchedule::GITS,
            Schedule::N_SCHEDULES => GSchedule::NSchedules,
            _ => GSchedule::Default,
        }
    }
}

impl Into<Schedule> for GSchedule {
    fn into(self) -> Schedule {
        match self {
            GSchedule::Default => Schedule::DEFAULT,
            GSchedule::Discrete => Schedule::DISCRETE,
            GSchedule::Karras => Schedule::KARRAS,
            GSchedule::Exponential => Schedule::EXPONENTIAL,
            GSchedule::AlignYourSteps => Schedule::AYS,
            GSchedule::GITS => Schedule::GITS,
            GSchedule::NSchedules => Schedule::N_SCHEDULES,
        }
    }
}

#[derive(GodotConvert, Debug, Var, Export, Copy, Clone, Default)]
#[godot(via = GString)]
pub enum GSampleMethod {
    EulerA,
    #[default]
    Euler,
    Heun,
    Dpm2,
    Dpmpp2sA,
    Dpmpp2m,
    Dpmpp2mv2,
    Ipndm,
    IpndmV,
    Lcm,
    NSampleMethods,
    DdimTrailing,
    Tcd,
}

impl From<SampleMethod> for GSampleMethod {
    fn from(sample_method: SampleMethod) -> Self {
        match sample_method {
            SampleMethod::EULER_A => GSampleMethod::EulerA,
            SampleMethod::EULER => GSampleMethod::Euler,
            SampleMethod::HEUN => GSampleMethod::Heun,
            SampleMethod::DPM2 => GSampleMethod::Dpm2,
            SampleMethod::DPMPP2S_A => GSampleMethod::Dpmpp2sA,
            SampleMethod::DPMPP2M => GSampleMethod::Dpmpp2m,
            SampleMethod::DPMPP2Mv2 => GSampleMethod::Dpmpp2mv2,
            SampleMethod::IPNDM => GSampleMethod::Ipndm,
            SampleMethod::IPNDM_V => GSampleMethod::IpndmV,
            SampleMethod::LCM => GSampleMethod::Lcm,
            SampleMethod::N_SAMPLE_METHODS => GSampleMethod::NSampleMethods,
            SampleMethod::DDIM_TRAILING => GSampleMethod::DdimTrailing,
            SampleMethod::TCD => GSampleMethod::Tcd,
            _ => GSampleMethod::Euler,
        }
    }
}

impl Into<SampleMethod> for GSampleMethod {
    fn into(self) -> SampleMethod {
        match self {
            GSampleMethod::EulerA => SampleMethod::EULER_A,
            GSampleMethod::Euler => SampleMethod::EULER,
            GSampleMethod::Heun => SampleMethod::HEUN,
            GSampleMethod::Dpm2 => SampleMethod::DPM2,
            GSampleMethod::Dpmpp2sA => SampleMethod::DPMPP2S_A,
            GSampleMethod::Dpmpp2m => SampleMethod::DPMPP2M,
            GSampleMethod::Dpmpp2mv2 => SampleMethod::DPMPP2Mv2,
            GSampleMethod::Ipndm => SampleMethod::IPNDM,
            GSampleMethod::IpndmV => SampleMethod::IPNDM_V,
            GSampleMethod::Lcm => SampleMethod::LCM,
            GSampleMethod::NSampleMethods => SampleMethod::N_SAMPLE_METHODS,
            GSampleMethod::DdimTrailing => SampleMethod::DDIM_TRAILING,
            GSampleMethod::Tcd => SampleMethod::TCD,
        }
    }
}

pub trait TryIntoRGB {
    fn try_into_rgb(&mut self) -> Result<RgbImage, Error>;
}

impl TryIntoRGB for Image {
    fn try_into_rgb(&mut self) -> Result<RgbImage, Error> {
        self.convert(Format::RGB8);
        let data = self.get_data();
        let width = self.get_width() as u32;
        let height = self.get_height() as u32;
        let image = match RgbImage::from_raw(width, height, data.to_vec()) {
            Some(image) => Ok(image),
            None => Err(Error::FAILED),
        };
        return image;
    }
}

pub trait IntoGImage {
    fn into_g_image(&mut self) -> Gd<Image>;
}

impl IntoGImage for RgbImage {
    fn into_g_image(&mut self) -> Gd<Image> {
        let data = PackedByteArray::from(self.as_ref());

        return Image::create_from_data(
            self.width() as i32,
            self.height() as i32,
            false,
            Format::RGB8,
            &data,
        )
        .unwrap();
    }
}
