# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

r"""Add HuggingFace Datasets.

Script to convert code for dataset in HuggingFace to tensorflow datasets

"""

import argparse
import os
import pathlib
import re
from typing import Optional

from absl import app
from absl import flags
from absl.flags import argparse_flags
import tensorflow.compat.v2 as tf
from tensorflow_datasets.core.utils import py_utils
from tensorflow_datasets.scripts.cli import new

# In TF 2.0, eager execution is enabled by default
tf.compat.v1.disable_eager_execution()

flags.DEFINE_string("tfds_dir", py_utils.tfds_dir(),
                    "Path to tensorflow_datasets directory")
FLAGS = flags.FLAGS


TO_CONVERT = [
    # (pattern, replacement)
    # Order is important here for some replacements
    (r"from\s__future.*", r""),
    (r"import\slogging", r"from absl import logging\n"),
    (r"import\snlp",
     r"import tensorflow as tf\nimport tensorflow_datasets.public_api as tfds\n"
    ),
    (r"with\sopen", r"with tf.io.gfile.GFile"),
    (r"nlp\.Value\(\"string\"\)", r"tfds.features.Text()"),
    (r"nlp\.Value\(\"string\"\),", r"tfds.features.Text("),
    (r"nlp\.Value\(\"([\w\d]+)\"\)", r"tf.\1"),
    (r"nlp\.features", "tfds.features"),
    (r"features\s*=\s*nlp\.Features\(",
     r"features=tfds.features.FeaturesDict("),
    (r"dict\(", r"tfds.features.FeaturesDict("),
    (r"nlp.SplitGenerator", r"tfds.core.SplitGenerator"),
    (r"self\.config\.data_dir", r"dl_manager.manual_dir"),
    (r"self\.config", r"self.builder_config"),
    (r"nlp\.Split", r"tfds.Split"),
    (r"nlp", r"tfds.core"),
]

PATTERNS = [
    # (pattern, replacement)
    # Order is important here for some replacements
    (r"from\s__future.*"),
    (r"import\slogging"),
    (r"import\snlp"),
    (r"with\sopen"),
    (r"nlp\.Value\(\"string\"\)"),
    (r"nlp\.Value\(\"string\"\),"),
    (r"nlp\.Value\(\"([\w\d]+)\"\)"),
    (r"nlp\.features"),
    (r"features\s*=\s*nlp\.Features\("),
    (r"dict\("),
    (r"nlp.SplitGenerator"),
    (r"self\.config\.data_dir"),
    (r"self\.config"),
    (r"nlp\.Split"),
    (r"nlp"),
]

COMPILED_PATTERNS = [
    (re.compile(pattern), replacement) for pattern, replacement in TO_CONVERT
]


def _hugging_face_path():
  return os.path.join(FLAGS.tfds_dir, "scripts", "cli", "data",
                      "huggingface.py")


def _dataset_dir(library, dataset_name):
  return os.path.join(FLAGS.tfds_dir, library, dataset_name)


def _parse_flags(_) -> argparse.Namespace:
  """Command line flags."""
  parser = argparse_flags.ArgumentParser(
      prog="convert_dataset",
      description="Tool to add hugging face datasets",
  )
  parser.add_argument(
      "--dataset_name",
      help="Path of hugging face.",
  )
  parser.add_argument(
      "--library",
      help="Path of tensorflow",
  )
  return parser.parse_args()


def main(args: argparse.Namespace):

  convert_dataset(
      dataset_name=args.dataset_name, library=args.library)


def convert_dataset(dataset_name: Optional[str] = None,
                    library: Optional[str] = None) -> None:
  """Conver Hugging Face Datasets."""

  compiled_regex = re.compile("|".join(PATTERNS))

  root_dir = pathlib.Path(_dataset_dir(library, dataset_name))
  root_dir.mkdir(parents=True)

  output_file = root_dir / "{}.py".format(dataset_name)
  input_file = _hugging_face_path()

  info = new.DatasetInfo(name=dataset_name, in_tfds=True, path=root_dir)
  new.create_dataset_test(info)
  new.create_init(info)
  new.create_dummy_data(info)
  new.create_checksum(info)
  new.create_build(info)

  with tf.io.gfile.GFile(input_file, "r") as f:
    lines = f.readlines()

    out_lines = []
    for line in lines:
      out_line = line

      if "return nlp.DatasetInfo(" in out_line:
        out_lines.append("    return tfds.core.DatasetInfo(\n")
        out_lines.append("        builder=self,\n")
        out_line = ""
      else:
        for compiled_regex, replacement in COMPILED_PATTERNS:
          out_line = compiled_regex.sub(replacement, out_line)

      assert ("nlp" not in out_line), f"Error converting {out_line.strip()}"
      out_lines.append(out_line)

      with open(output_file, "w") as f:
        f.writelines(out_lines)


if __name__ == "__main__":
  app.run(main, flags_parser=_parse_flags)
