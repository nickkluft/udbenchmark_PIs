# UDBENCHMARK

**TODO**: general description

## Install

```term
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install -r PI_udbenchmark_github/src/pi_FPE/requirements.txt
pip install -e PI_udbenchmark_github/src/pi_FPE

pip install -r PI_udbenchmark_github/src/pi_FPM/requirements.txt
pip install -e PI_udbenchmark_github/src/pi_FPM

pip install -r PI_udbenchmark_github/src/pi_LDE/requirements.txt
pip install -e PI_udbenchmark_github/src/pi_LDE

pip install -r PI_udbenchmark_github/src/pi_SpatTemp/requirements.txt
pip install -e PI_udbenchmark_github/src/pi_SpatTemp

```

## Usage

```term
run_fpm PI_udbenchmark_github/tests/data/input/subject_04_jointTrajectories_01.csv PI_udbenchmark_github/tests/data/input/subject_04_comTrajectories_01.csv PI_udbenchmark_github/tests/data/input/subject_04_gaitEvents_01 outfolder
```

```term
run_spattemp PI_udbenchmark_github/tests/data/input/subject_04_jointTrajectories_01.csv PI_udbenchmark_github/tests/data/input/subject_04_gaitEvents_01 outfolder
```

```term
run_lde PI_udbenchmark_github/tests/data/input/subject_04_comTrajectories_01.csv PI_udbenchmark_github/tests/data/input/subject_04_gaitEvents_01 outfolder
```
