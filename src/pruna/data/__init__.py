# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple

from pruna.data.datasets.audio import (
    setup_librispeech_dataset,
    setup_mini_presentation_audio_dataset,
    setup_podcast_dataset,
)
from pruna.data.datasets.image import (
    setup_cifar10_dataset,
    setup_imagenet_dataset,
    setup_mnist_dataset,
)
from pruna.data.datasets.prompt import (
    setup_dpg_dataset,
    setup_drawbench_dataset,
    setup_gedit_dataset,
    setup_genai_bench_dataset,
    setup_geneval_dataset,
    setup_hps_dataset,
    setup_imgedit_dataset,
    setup_long_text_bench_dataset,
    setup_oneig_dataset,
    setup_parti_prompts_dataset,
)
from pruna.data.datasets.question_answering import setup_polyglot_dataset
from pruna.data.datasets.text_generation import (
    setup_c4_dataset,
    setup_openassistant_dataset,
    setup_pubchem_dataset,
    setup_smolsmoltalk_dataset,
    setup_smoltalk_dataset,
    setup_tiny_imdb_dataset,
    setup_wikitext_dataset,
    setup_wikitext_tiny_dataset,
)
from pruna.data.datasets.text_to_image import (
    setup_coco_dataset,
    setup_laion256_dataset,
    setup_open_image_dataset,
)
from pruna.data.datasets.text_to_video import setup_vbench_dataset

base_datasets: dict[str, Tuple[Callable, str, dict[str, Any]]] = {
    "COCO": (setup_coco_dataset, "image_generation_collate", {"img_size": 512}),
    "LAION256": (setup_laion256_dataset, "image_generation_collate", {"img_size": 512}),
    "LibriSpeech": (setup_librispeech_dataset, "audio_collate", {}),
    "AIPodcast": (setup_podcast_dataset, "audio_collate", {}),
    "MiniPresentation": (setup_mini_presentation_audio_dataset, "audio_collate", {}),
    "ImageNet": (
        setup_imagenet_dataset,
        "image_classification_collate",
        {"img_size": 224},
    ),
    "MNIST": (setup_mnist_dataset, "image_classification_collate", {"img_size": 28}),
    "WikiText": (setup_wikitext_dataset, "text_generation_collate", {}),
    "TinyWikiText": (setup_wikitext_tiny_dataset, "text_generation_collate", {}),
    "SmolTalk": (setup_smoltalk_dataset, "text_generation_collate", {}),
    "SmolSmolTalk": (setup_smolsmoltalk_dataset, "text_generation_collate", {}),
    "PubChem": (setup_pubchem_dataset, "text_generation_collate", {}),
    "OpenAssistant": (setup_openassistant_dataset, "text_generation_collate", {}),
    "C4": (setup_c4_dataset, "text_generation_collate", {}),
    "Polyglot": (setup_polyglot_dataset, "question_answering_collate", {}),
    "OpenImage": (
        setup_open_image_dataset,
        "image_generation_collate",
        {"img_size": 1024},
    ),
    "CIFAR10": (
        setup_cifar10_dataset,
        "image_classification_collate",
        {"img_size": 32},
    ),
    # our full CIFAR10 has 50k train and 10k test
    "TinyCIFAR10": (
        partial(setup_cifar10_dataset, train_sample_size=800, test_sample_size=100),
        "image_classification_collate",
        {"img_size": 32},
    ),
    #  our full MNIST has 60k train and 10k test
    "TinyMNIST": (
        partial(setup_mnist_dataset, train_sample_size=800, test_sample_size=100),
        "image_classification_collate",
        {"img_size": 28},
    ),
    # our full ImageNet has 100k train and 10k val
    "TinyImageNet": (
        partial(setup_imagenet_dataset, train_sample_size=1000, test_sample_size=100),
        "image_classification_collate",
        {"img_size": 224},
    ),
    "DrawBench": (setup_drawbench_dataset, "prompt_collate", {}),
    "PartiPrompts": (
        setup_parti_prompts_dataset,
        "prompt_with_auxiliaries_collate",
        {},
    ),
    "GenAIBench": (setup_genai_bench_dataset, "prompt_collate", {}),
    "GenEval": (setup_geneval_dataset, "prompt_with_auxiliaries_collate", {}),
    "HPS": (setup_hps_dataset, "prompt_with_auxiliaries_collate", {}),
    "ImgEdit": (setup_imgedit_dataset, "prompt_with_auxiliaries_collate", {}),
    "LongTextBench": (setup_long_text_bench_dataset, "prompt_with_auxiliaries_collate", {}),
    "GEditBench": (setup_gedit_dataset, "prompt_with_auxiliaries_collate", {}),
    "OneIG": (setup_oneig_dataset, "prompt_with_auxiliaries_collate", {}),
    "DPG": (setup_dpg_dataset, "prompt_with_auxiliaries_collate", {}),
    "TinyIMDB": (setup_tiny_imdb_dataset, "text_generation_collate", {}),
    "VBench": (setup_vbench_dataset, "prompt_with_auxiliaries_collate", {}),
}


@dataclass
class Benchmark:
    """
    Metadata for a benchmark dataset.

    Parameters
    ----------
    name : str
        Internal identifier for the benchmark.
    display_name : str
        Human-readable name for display purposes.
    description : str
        Description of what the benchmark evaluates.
    metrics : list[str]
        List of metric names used for evaluation.
    task_type : str
        Type of task the benchmark evaluates (e.g., 'text_to_image').
    """

    name: str
    display_name: str
    description: str
    metrics: list[str]
    task_type: str


benchmark_info: dict[str, Benchmark] = {
    "PartiPrompts": Benchmark(
        name="parti_prompts",
        display_name="Parti Prompts",
        description=(
            "Holistic benchmark from Google Research with over 1,600 English prompts across 12 categories "
            "and 11 challenge aspects. Evaluates text-to-image models on abstract thinking, world knowledge, "
            "perspectives, and symbol rendering from basic to complex compositions."
        ),
        metrics=["arniqa", "clip_score", "clipiqa", "sharpness"],
        task_type="text_to_image",
    ),
    "DrawBench": Benchmark(
        name="drawbench",
        display_name="DrawBench",
        description=(
            "Comprehensive benchmark from the Imagen team for rigorous evaluation of text-to-image models. "
            "Enables side-by-side comparison on sample quality and image-text alignment with human raters."
        ),
        metrics=[
            "clip_score",
            "clipiqa",
            "sharpness",
            # "image_reward" not supported in Pruna
        ],
        task_type="text_to_image",
    ),
    "GenAIBench": Benchmark(
        name="genai_bench",
        display_name="GenAI Bench",
        description=(
            "1,600 prompts from professional designers for compositional text-to-visual generation. "
            "Covers basic skills (scene, attributes, spatial relationships) to advanced reasoning "
            "(counting, comparison, logic/negation) with over 24k human ratings."
        ),
        metrics=[
            "clip_score",
            "clipiqa",
            "sharpness",
            # "vqa" not supported in Pruna
        ],
        task_type="text_to_image",
    ),
    "VBench": Benchmark(
        name="vbench",
        display_name="VBench",
        description=(
            "Comprehensive benchmark suite for video generative models. Decomposes video quality into "
            "16 disentangled dimensions: temporal flickering, motion smoothness, subject consistency, "
            "spatial relationship, color, aesthetic quality, and more."
        ),
        metrics=["clip_score"],
        task_type="text_to_video",
    ),
    "GenEval": Benchmark(
        name="geneval",
        display_name="GenEval",
        description=(
            "Object-focused framework (NeurIPS 2023) for fine-grained text-to-image alignment. "
            "Evaluates compositional properties: object co-occurrence, position, count, and color binding "
            "via instance-level analysis rather than distribution-level metrics."
        ),
        metrics=[
            # "qa_accuracy" not supported in Pruna
        ],
        task_type="text_to_image",
    ),
    "HPS": Benchmark(
        name="hps",
        display_name="HPS",
        description=(
            "Human Preference Score v2: large-scale benchmark with 798k human preference choices on "
            "433k image pairs. CLIP fine-tuned on HPD v2 to predict human preferences and align "
            "evaluation with actual human judgment across diverse generative outputs."
        ),
        metrics=[
            # "hps" not supported in Pruna
        ],
        task_type="text_to_image",
    ),
    "LongTextBench": Benchmark(
        name="long_text_bench",
        display_name="Long Text Bench",
        description=(
            "DetailMaster benchmark with prompts averaging 284.89 tokens. Evaluates four dimensions: "
            "character attributes, structured locations, scene attributes, and spatial relationships "
            "to test compositional reasoning under long prompt complexity."
        ),
        metrics=[
            # "text_score" not supported in Pruna
        ],
        task_type="text_to_image",
    ),
    "ImgEdit": Benchmark(
        name="imgedit",
        display_name="ImgEdit",
        description=(
            "Unified image editing benchmark (PKU-YuanGroup) with 8 edit types: replace, add, remove, "
            "adjust, extract, style, background, compose. Evaluates instruction adherence, editing "
            "quality, and detail preservation."
        ),
        metrics=[
            # "img_edit_score" not supported in Pruna
        ],
        task_type="image_edit",
    ),
    "GEditBench": Benchmark(
        name="gedit_bench",
        display_name="GEdit Bench",
        description=(
            "StepFun benchmark grounded in real-world user instructions. 11 task types including "
            "background_change, subject_add/remove/replace, style_change, and tone_transfer for "
            "practical evaluation of image editing capabilities."
        ),
        metrics=[
            # "viescore" not supported in Pruna
        ],
        task_type="image_edit",
    ),
    "OneIG": Benchmark(
        name="oneig",
        display_name="OneIG",
        description=(
            "Omni-dimensional benchmark (NeurIPS 2025) for nuanced image generation evaluation. "
            "Six categories: Text_Rendering, Anime_Stylization, Portrait, General_Object, "
            "Knowledge_Reasoning, Multilingualism. Addresses text rendering precision and prompt-image alignment."
        ),
        metrics=[
            # "alignment_score", "text_score" not supported in Pruna
        ],
        task_type="text_to_image",
    ),
    "DPG": Benchmark(
        name="dpg",
        display_name="DPG",
        description=(
            "Dense Prompt Graph benchmark from ELLA/Tencent. ~1,000 complex prompts testing "
            "entity, attribute, relation, and global aspects. Evaluates models on dense prompt "
            "following with multiple objects and varied attributes."
        ),
        metrics=[
            # "qa_accuracy" not supported in Pruna
        ],
        task_type="text_to_image",
    ),
    "COCO": Benchmark(
        name="coco",
        display_name="COCO",
        description=(
            "Microsoft COCO dataset for image generation evaluation. Real image-caption pairs "
            "enabling FID and alignment metrics on distribution-level and instance-level quality."
        ),
        metrics=["fid", "clip_score", "clipiqa"],
        task_type="text_to_image",
    ),
    "ImageNet": Benchmark(
        name="imagenet",
        display_name="ImageNet",
        description=(
            "Large-scale image classification benchmark with 1,000 classes. Standard evaluation "
            "for vision model accuracy on object recognition."
        ),
        metrics=["accuracy"],
        task_type="image_classification",
    ),
    "WikiText": Benchmark(
        name="wikitext",
        display_name="WikiText",
        description=(
            "Language modeling benchmark based on Wikipedia articles. Standard evaluation "
            "for text generation quality via perplexity."
        ),
        metrics=["perplexity"],
        task_type="text_generation",
    ),
}
