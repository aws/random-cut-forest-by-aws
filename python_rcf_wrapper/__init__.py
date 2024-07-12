from pathlib import Path

import logging

# Import JPype Module for Java imports
import jpype.imports
from jpype.types import *

import os

java_home = os.environ.get("JAVA_HOME", None)

DEFAULT_JAVA_PATH = Path(__file__).parent / "lib"


java_path = str(Path(os.environ.get("JAVA_LIB", DEFAULT_JAVA_PATH)) / "*")

jpype.addClassPath(java_path)

# Launch the JVM
jpype.startJVM(convertStrings=False)

logging.info("availableProcess {}".format(jpype.java.lang.Runtime.getRuntime().availableProcessors()))
