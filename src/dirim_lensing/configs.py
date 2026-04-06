from pydantic import BaseModel, Field
from typing import Literal, Optional

class DatasetNameConfig(BaseModel):
    name: Literal["SKIRT_EPL", "SKIRT_TNG"] = Field(description="(string in {'SKIRT_EPL','SKIRT_TNG'}): Name of the dataset to be used")
    res: int                                = Field(description="(int): Resolution of the images (number of pixels per side)")

class SkirtEPLDatasetConfig(BaseModel):
    save_path:             str           = Field(description="(string): Path where to save dataset")
    train_split:           float         = Field(description="(float in [0,1]): Fraction of dataset dedicated to training")
    validation_split:      float         = Field(description="(float in [0,1]): Fraction of dataset dedicated to validation")
    test_split:            float         = Field(description="(float in [0,1]): Fraction of dataset dedicated to testing")
    sigma_y :              float         = Field(description="(float): Standard deviation of Gaussian observational noise")
    pixelscale:            float         = Field(description="(float): Angular scale of one pixel in the lens plane [arcsec/pixel]")
    source_pixelscale:     float         = Field(description="(float): Angular scale of one pixel in the source plane [arcsec/pixel]")
    z_l:                   float         = Field(description="(float): Redshift of lens")
    z_s:                   float         = Field(description="(float): Redshift of source")
    psf_sigma:             float         = Field(description="(float): Standard deviation of Gaussian PSF [arcsec]")
    upsample_factor:       int           = Field(description="(int >= 1): Upsampling factor for the PSF and the lensed image")
    upsample_factor_kappa: int           = Field(description="(int >= 1): Upsampling factor for the kappa maps")
    quad_level_kappa:      Optional[int] = Field(description="(int >= 2 or none): Gaussian quadrature level for kappa map upsampling")
    x0_min:                float         = Field(description="(float): Lower bound for the x-position of the center of the lenses [arcsec]")
    x0_max:                float         = Field(description="(float): Upper bound for the x-position of the center of the lenses [arcsec]")
    y0_min:                float         = Field(description="(float): Lower bound for the y-position of the center of the lenses [arcsec]")
    y0_max:                float         = Field(description="(float): Upper bound for the y-position of the center of the lenses [arcsec]")
    q_min:                 float         = Field(description="(float): Lower bound for axis ratio of lenses")
    q_max:                 float         = Field(description="(float): Upper bound for axis ratio of lenses")
    phi_min:               float         = Field(description="(float in [0,pi]): Lower bound for angular position of lenses [radian]")
    phi_max:               float         = Field(description="(float in [0,pi]): Upper bound for angular position of lenses [radian]")
    Rein_min:              float         = Field(description="(float): Lower bound for Einstein radius of lenses [arcsec]")
    Rein_max:              float         = Field(description="(float): Upper bound for Einstein radius of lenses [arcsec]")
    tau_min:               float         = Field(description="(float): Lower bound for EPL power law slope for lenses")
    tau_max:               float         = Field(description="(float): Upper bound for EPL power law slope for lenses")
    am3_min:               float         = Field(description="(float): Lower bound for m=3 multipole amplitude [1/arcsec]")
    am3_max:               float         = Field(description="(float): Upper bound for m=3 multipole amplitude [1/arcsec]")
    am4_min:               float         = Field(description="(float): Lower bound for m=4 multipole amplitude [1/arcsec]")
    am4_max:               float         = Field(description="(float): Upper bound for m=4 multipole amplitude [1/arcsec]")
    thetam3_min:           float         = Field(description="(float in [0,2pi]): Lower bound for m=3 multipole angular position [radians]")
    thetam3_max:           float         = Field(description="(float in [0,2pi]): Upper bound for m=3 multipole angular position [radians]")
    thetam4_min:           float         = Field(description="(float in [0,2pi]): Lower bound for m=4 multipole angular position [radians]")
    thetam4_max:           float         = Field(description="(float in [0,2pi]): Upper bound for m=4 multipole angular position [radians]")
    logM_min:              float         = Field(description="(float): Lower bound for log_10 of the subhalo masses [log_10 M_sun]")
    logM_max:              float         = Field(description="(float): Upper bound for log_10 of the subhalo masses [log_10 M_sun]")
    c_min:                 float         = Field(description="(float): Lower bound for concentration factor of NFW subhalos")
    c_max:                 float         = Field(description="(float): Upper bound for concentration factor of NFW subhalos")
    augment:               bool          = Field(description="(bool): Whether to do source data augmentation (rotations and flips)")

class SkirtTNGDatasetConfig(BaseModel):
    save_path:         str   = Field(description="(string): Path where to save dataset")
    train_split:       float = Field(description="(float in [0,1]): Fraction of dataset dedicated to training")
    validation_split:  float = Field(description="(float in [0,1]): Fraction of dataset dedicated to validation")
    test_split:        float = Field(description="(float in [0,1]): Fraction of dataset dedicated to testing")
    sigma_y :          float = Field(description="(float): Standard deviation of Gaussian observational noise")
    pixelscale:        float = Field(description="(float): Angular scale of one pixel in the lens plane [arcsec/pixel]")
    source_pixelscale: float = Field(description="(float): Angular scale of one pixel in the source plane [arcsec/pixel]")
    z_l:               float = Field(description="(float): Redshift of lens")
    z_s:               float = Field(description="(float): Redshift of source")
    psf_sigma:         float = Field(description="(float): Standard deviation of Gaussian PSF [arcsec]")
    upsample_factor:   int   = Field(description="(int >= 1): Upsampling factor for the PSF and the lensed image")
    augment:           bool  = Field(description="(bool): Whether to do source data augmentation (rotations and flips)")

class SDEConfig(BaseModel):
    space_kappa: Literal["linear","log"]              = Field(description="(string in {'linear','log'}): Space in which diffusion noise is added to kappa maps")
    kind:        Literal["VE","VP_linear","VP_exp"]   = Field(description="(string in {'VE','VP_linear','VP_exp'}): Choice of SDE (Variance exploding or Variance preserving) and noise schedule (linear or exponential)")
    sigma_min:   Optional[float]                      = Field(description="(float or none): Minimum noise scale in VE SDE")
    sigma_max:   Optional[float]                      = Field(description="(float or none): Maximum noise scale in VE SDE")
    beta_min:    Optional[float]                      = Field(description="(float or none): Minimum noise scale in VP SDE")
    beta_max:    Optional[float]                      = Field(description="(float or none): Maximum noise scale in VP SDE")
    epsilon:     float                                = Field(description="(float in [0,1]): The diffusion process is limited to range [epsilon,1]")

class RIMConfig(BaseModel):
    num_iterations:     int                                   = Field(description="(int): Number of iterations of the RIM at training and inference time")
    model_channels:     int                                   = Field(description="(int): Number of convolutional filters in first UNet level")
    channel_mult:       list[int]                             = Field(description="(list): Multiplier for number of convolutional filters across UNet levels")
    num_blocks:         int                                   = Field(description="(int): Number of residual blocks per UNet level")
    attn_resolutions:   list[int]                             = Field(description="(list): Resolutions at which self-attention is applied")
    embedding_type:     Literal["positional","fourier"]       = Field(description="(string in {'positional','fourier'}): Timestep embedding type")
    channel_mult_noise: int                                   = Field(description="(int): Timestep embedding size multiplier")
    encoder_type:       Literal["standard","residual","skip"] = Field(description="(string in {'standard','residual','skip'}): Encoder architecture")
    decoder_type:       Literal["standard","skip"]            = Field(description="(string in {'standard','skip'}): Decoder architecture")
    resample_filter:    list[float]                           = Field(description="(list): Resampling filter")
    use_residuals:      bool                                  = Field(description="(bool): Whether to take lensed image reconstruction residuals as additional input channel")
    use_log_t:          bool                                  = Field(description="(bool): Whether to take log of diffusion time as input to the model")
     
    class GradLikConfig(BaseModel):
        type:             Literal["tanh","arcsinh","Adam"] = Field(description="(string in {'tanh','arcsinh','Adam'}): Type of likelihood gradient pre-processing")
        grad_norm_source: Optional[float]                  = Field(description="(float or none): Source gradient normalization value for tanh or arcsinh type")
        grad_norm_kappa:  Optional[float]                  = Field(description="(float or none): Kappa map gradient normalization value for tanh or arcsinh type")
        adam_epsilon:     Optional[float]                  = Field(description="(float or none): epsilon in Adam optimizer")
        beta1:            Optional[float]                  = Field(description="(float or none): beta1 in Adam optimizer")
        beta2:            Optional[float]                  = Field(description="(float or none): beta2 in Adam optimizer")

    grad_lik: GradLikConfig

    class MemoryConfig(BaseModel):
        type: Optional[Literal["GRU","residual"]] = Field(description="(false or string in {'GRU','residual'}): Type of memory to use")

    memory: MemoryConfig

class TrainingConfig(BaseModel):
    batch_size:         int   = Field(description="(int): Training batch size")
    learning_rate:      float = Field(description="(float): Training initial learning rate")
    lr_decay:           float = Field(description="(float): Learning rate exponential decay rate per epoch")
    dropout:            float = Field(description="(float in [0,1]): Dropout rate")
    gradient_clipping:  float = Field(description="(float): Maximum gradient norm above which clipping is applied")
    ema_decay:          float = Field(description="(float in [0,1]): Exponential moving average decay rate for weights")
    num_epochs:         int   = Field(description="(int): Number of epochs to train for")
    patience:           int   = Field(description="(int): Number of consecutive epochs without validation loss improvement needed to trigger early stopping")
    resume_train:       bool  = Field(description="(bool): Whether to continue training an existing model")
    start_epoch:        int   = Field(description="(int): Epoch at which to start training")

class LossConfig(BaseModel):
    time_weights_cutoff:  float                           = Field(description="(float): Cutoff for time weights in loss function")
    kappa_weights:        Literal["uniform","sqrt_kappa"] = Field(description="(string in {'uniform','sqrt_kappa'}): Kappa map pixel weights in loss function")
    iteration_weights:    str | list[float]               = Field(description="('uniform' or list of floats): Relative weight of RIM iterations in loss function")

class SamplingConfig(BaseModel):
    model_epoch:  int                                  = Field(description="(int): Number of epochs of training for the model to be used")
    sampler_name: Literal["EM",'Heun','Euler','RK4']   = Field(description="(string in {'EM, Heun, Euler, RK4'}): Method for solving the reverse SDE")
    num_steps:    int                                  = Field(description="(int): Number of time steps to take when solving the reverse SDE")
    jump_to_0:    bool                                 = Field(description="(bool): Whether to denoise the samples using the denoising model upon reaching t=epsilon")
    n_corrector:  int                                  = Field(description="(int): Number of corrector steps per predictor step")
    snr:          Optional[float]                      = Field(description="(float): Signal to noise ratio for corrector steps")

class TARPConfig(BaseModel):
    n_samples:       int              = Field(description="(int): Number of samples to generate per simulation")
    n_sims:          int              = Field(description="(int): Number of simulations to process")
    num_alpha_bins:  int              = Field(description="(int): Number of alpha bins in TARP plot")
    norm:            bool             = Field(description="(bool): Whether to normalize space in TARP plot")
    references:      Literal["prior"] = Field(description="(string in {'prior'}): Distribution to use to generate reference points")

class TestsConfig(BaseModel):
    sample_idxs: list[int] = Field(description="(list of ints): Indices of samples in the test set to run tests on")
    n_samples:   int       = Field(description="(int): Number of samples to generate per test simulation")
    batch_size:  int       = Field(description="(int): Batch size to use at sampling time when running tests")

class Config(BaseModel):
    dataset: Optional[DatasetNameConfig]
    skirt_epl_dataset: Optional[SkirtEPLDatasetConfig]
    skirt_tng_dataset: Optional[SkirtTNGDatasetConfig]
    sde: Optional[SDEConfig]
    rim: Optional[RIMConfig]
    training: Optional[TrainingConfig]
    loss: Optional[LossConfig]
    sampling: Optional[SamplingConfig]
    tarp: Optional[TARPConfig]
    tests: Optional[TestsConfig]