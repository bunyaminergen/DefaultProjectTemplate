import logging
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, conint, confloat, constr


# Teacher
class PipelineConfig(BaseModel):
    name: str = "text-generation"
    load_in_4bit: bool = True
    max_length: int = 2048
    temperature: float = 0.3
    truncation: bool = True
    do_sample: bool = True


class TeacherConfig(BaseModel):
    model: str
    save_path: str
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    pipeline: PipelineConfig = PipelineConfig()


# CUDA
class CudaConfig(BaseModel):
    cuda_alloc_conf: str = "expandable_segments:True"


# Device
class DeviceConfig(BaseModel):
    type: Literal["cuda", "cpu", "auto"] = "cuda"


# Dataset
class DatasetItem(BaseModel):
    name: str
    source: Literal["huggingface"]
    key: str
    subset: Optional[str] = "default"


# Assistant
class AssistantModelConfig(BaseModel):
    max_epochs: conint(gt=0)
    batch_size: conint(gt=0)
    learning_rate: confloat(gt=0.0)
    max_seq_length: conint(gt=0)
    cutoff_len: conint(gt=0)
    lora_r: conint(gt=0)
    lora_alpha: conint(gt=0)
    lora_dropout: confloat(ge=0.0, le=1.0)


class AssistantsConfig(BaseModel):
    gpt2: AssistantModelConfig


# RootConfig
class RootConfig(BaseModel):
    cuda: CudaConfig = CudaConfig()
    device: DeviceConfig = DeviceConfig()
    teacher: TeacherConfig
    assistants: AssistantsConfig
    datasets: List[DatasetItem]


# Logger
class LoggerType(BaseModel):
    name: constr(min_length=1) = Field(..., description="Logger name")
    app: constr(min_length=1) = Field("Ayvaz", description="Application name")
    path: constr(min_length=1) = Field(".logs", description="The folder in which log files are stored")
    file: constr(min_length=1) = Field(".log", description="Name of the log file")
    console_level: conint(ge=0) = Field(logging.DEBUG, description="Console log level")
    file_level: conint(ge=0) = Field(logging.DEBUG, description="File log level")
    max_bytes: conint(gt=0) = Field(5_000_000, description="Maximum file size for rotation")
    backup_count: conint(gt=0) = Field(5, description="Number of old logs to keep")
    verbose: bool = Field(True, description="Whether to output logs to the console")


class SingleLineConsoleFormatterType(BaseModel):
    app: constr(min_length=1) = Field("Ayvaz", description="Application name")
    date_format: Optional[str] = Field(None, description="Date format")


class SingleLineFileFormatterType(BaseModel):
    app: constr(min_length=1) = Field("Ayvaz", description="Application name")
    date_format: Optional[str] = Field(None, description="Date format")
