import logging
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, conint, confloat, constr, StrictBool


# Teacher
class TeacherModelConfig(BaseModel):
    name: constr(min_length=1) = Field(..., description="Hugging Face model name/path for the teacher.")
    save_path: constr(min_length=1) = Field(..., description="Local directory path where the teacher model is/will be s"
                                                             "aved.")
    load_in_4bit: StrictBool = Field(True, description="Whether to load the teacher model in 4-bit precision.")
    bnb_4bit_use_double_quant: StrictBool = Field(True, description="Enable double-quantization in BitsAndBytes 4-bit m"
                                                                    "ode.")
    bnb_4bit_quant_type: constr(min_length=1) = Field("nf4", description="Type of quantization (e.g., 'nf4') used by Bi"
                                                                         "tsAndBytes.")


class TeacherPipelineConfig(BaseModel):
    name: constr(min_length=1) = Field("text-generation", description="Name of the pipeline, e.g., 'text-generation'.")
    load_in_4bit: StrictBool = Field(True, description="Whether to load the pipeline in 4-bit precision.")
    max_length: conint(ge=1) = Field(512, description="Maximum sequence length for teacher text generation.")
    temperature: confloat(ge=0.0, le=100.0) = Field(0.3, description="Sampling temperature; higher values produce more "
                                                                     "varied outputs.")
    truncation: StrictBool = Field(True, description="Whether to truncate sequences to 'max_length'.")
    do_sample: StrictBool = Field(True, description="Whether to sample the output (if False, uses greedy decoding).")
    batch_size: conint(ge=1) = Field(16, description="The number of instructions to process simultaneously during model"
                                                     " generation.")


class TeacherDataConfig(BaseModel):
    prompt_length: conint(gt=0) = Field(29, description="Maximum number of words in a prompt (to be processed).")
    output_path: constr(min_length=1) = Field(..., description="Output path for teacher-generated JSONL file.")


class TeacherConfig(BaseModel):
    model: TeacherModelConfig = Field(..., description="Model configuration for the teacher.")
    pipeline: TeacherPipelineConfig = Field(..., description="Pipeline configuration for the teacher.")
    data: TeacherDataConfig = Field(..., description="Data configuration for the teacher (prompt length, output path).")


# Student
class StudentModelConfig(BaseModel):
    name: constr(min_length=1) = Field(..., description="Hugging Face model name/path for the student.")
    save_path: constr(min_length=1) = Field(...,
                                            description="Local directory where the student model is/will be saved.")


class StudentConfig(BaseModel):
    model: StudentModelConfig = Field(..., description="Model configuration for the student.")


# Cuda
class CudaConfig(BaseModel):
    cuda_alloc_conf: constr(min_length=1) = Field("expandable_segments:True",
                                                  description="Configuration for CUDA memory allocation (e.g. 'expandab"
                                                              "le_segments:True').")


# Device
class DeviceConfig(BaseModel):
    type: Literal["cuda", "cpu", "auto"] = Field("cuda",
                                                 description="Device type to use: 'cuda' (GPU), 'cpu', or 'auto' (auto-"
                                                             "detect).")


# Dataset
class DatasetItem(BaseModel):
    name: constr(min_length=1) = Field(..., description="Hugging Face dataset name (e.g., 'username/dataset_name').")
    key: constr(min_length=1) = Field(..., description="Key inside the dataset from which to extract prompts.")
    subset: Optional[constr(min_length=1)] = Field("default",
                                                   description="Optional dataset subset or config name. Defaults to 'de"
                                                               "fault'.")


# Assistant
class AssistantModelConfig(BaseModel):
    max_epochs: conint(gt=0) = Field(..., description="Number of training epochs (must be > 0).")
    batch_size: conint(gt=0) = Field(..., description="Training batch size (must be > 0).")
    learning_rate: confloat(gt=0.0) = Field(..., description="Learning rate (must be > 0).")
    max_seq_length: conint(gt=0) = Field(..., description="Max sequence length for training (must be > 0).")
    cutoff_len: conint(gt=0) = Field(..., description="Cutoff length for sequences (must be > 0).")
    lora_r: conint(gt=0) = Field(..., description="LoRA rank dimension (must be > 0).")
    lora_alpha: conint(gt=0) = Field(..., description="LoRA alpha scaling factor (must be > 0).")
    lora_dropout: confloat(ge=0.0, le=1.0) = Field(..., description="LoRA dropout rate (between 0.0 and 1.0).")


class AssistantsConfig(BaseModel):
    gpt2: AssistantModelConfig = Field(..., description="Configuration for the GPT-2 assistant model.")


# Root
class RootConfig(BaseModel):
    cuda: CudaConfig = Field(default_factory=CudaConfig, description="CUDA-related configuration settings.")
    device: DeviceConfig = Field(default_factory=DeviceConfig,
                                 description="Specifies the device for computations (CPU, GPU, or auto).")
    teacher: TeacherConfig = Field(..., description="Configuration for the Teacher.")
    assistants: AssistantsConfig = Field(..., description="Configuration for assistant models.")
    student: StudentConfig = Field(..., description="Configuration for the Student.")
    datasets: List[DatasetItem] = Field(..., description="List of datasets to be downloaded and processed.")


# Logger
class LoggerType(BaseModel):
    name: constr(min_length=1) = Field(..., description="Name of the logger.")
    app: constr(min_length=1) = Field("Ayvaz", description="Name of the application.")
    path: constr(min_length=1) = Field(".logs", description="Folder where log files are stored.")
    file: constr(min_length=1) = Field(".log", description="Name of the log file.")
    console_level: conint(ge=0) = Field(logging.DEBUG, description="Console log level.")
    file_level: conint(ge=0) = Field(logging.DEBUG, description="File log level.")
    max_bytes: conint(gt=0) = Field(5_000_000, description="Maximum log file size in bytes before rotation.")
    backup_count: conint(gt=0) = Field(5, description="Number of old log files to keep.")
    verbose: StrictBool = Field(True, description="Whether to output logs to the console.")


class SingleLineConsoleFormatterType(BaseModel):
    app: constr(min_length=1) = Field("Ayvaz", description="Name of the application.")
    date_format: Optional[str] = Field(None, description="Date format for console logs.")


class SingleLineFileFormatterType(BaseModel):
    app: constr(min_length=1) = Field("Ayvaz", description="Name of the application.")
    date_format: Optional[str] = Field(None, description="Date format for file logs.")
