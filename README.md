# UDBENCHMARK

**TODO**: general description

## Install

```term
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install -r src/pi_FPE/requirements.txt
pip install -e src/pi_FPE

pip install -r src/pi_FPM/requirements.txt
pip install -e src/pi_FPM

pip install -r src/pi_LDE/requirements.txt
pip install -e src/pi_LDE

pip install -r src/pi_SpatTemp/requirements.txt
pip install -e src/pi_SpatTemp

```

## Usage

```term
run_fpe tests/data/input/subject_04_cond_23_run_01_jointTrajectories.csv tests/data/input/subject_04_cond_23_run_01_com.csv tests/data/input/subject_04_cond_23_run_01_angularMomentum.csv tests/data/input/subject_04_cond_23_run_01_inertiaTensor.csv tests/data/input/subject_04_cond_23_run_01_gaitEvents.yaml tests/data/input/condition_23.yaml out_tests
```

```term
run_fpm tests/data/input/subject_04_cond_23_run_01_jointTrajectories.csv tests/data/input/subject_04_cond_23_run_01_com.csv tests/data/input/subject_04_cond_23_run_01_gaitEvents.yaml out_tests
```

```term
run_spattemp tests/data/input/subject_04_cond_23_run_01_jointTrajectories.csv tests/data/input/subject_04_cond_23_run_01_gaitEvents.yaml out_tests
```

```term
run_lde tests/data/input/subject_04_cond_23_run_01_com.csv tests/data/input/subject_04_cond_23_run_01_gaitEvents.yaml out_tests
```

## Build docker image

_(only tested under linux)_

Run the following command in order to create the docker image for this PI:

```console
docker build . -t pi_udbenchmark
```

## Launch the docker image

Assuming the `tests/data/input` contains the input data, and that the directory `out_tests/` is **already created**, and will contain the PI output:

```shell
docker run --rm -v $PWD/tests/data/input:/in -v $PWD/out_tests:/out pi_udbenchmark run_fpe /in/subject_04_cond_23_run_01_jointTrajectories.csv /in/subject_04_cond_23_run_01_com.csv /in/subject_04_cond_23_run_01_angularMomentum.csv /in/subject_04_cond_23_run_01_inertiaTensor.csv /in/subject_04_cond_23_run_01_gaitEvents.yaml /in/condition_23.yaml /out
```

```shell
docker run --rm -v $PWD/tests/data/input:/in -v $PWD/out_tests:/out pi_udbenchmark run_fpm /in/subject_04_cond_23_run_01_jointTrajectories.csv /in/subject_04_cond_23_run_01_com.csv /in/subject_04_cond_23_run_01_gaitEvents.yaml /out
```

```shell
docker run --rm -v $PWD/tests/data/input:/in -v $PWD/out_tests:/out pi_udbenchmark run_spattemp /in/subject_04_cond_23_run_01_jointTrajectories.csv /in/subject_04_cond_23_run_01_gaitEvents.yaml /out
```

```shell
docker run --rm -v $PWD/tests/data/input:/in -v $PWD/out_tests:/out pi_udbenchmark run_lde /in/subject_04_cond_23_run_01_com.csv /in/subject_04_cond_23_run_01_gaitEvents.yaml /out
```

## Test the generate docker image

A generic testing process is proposed in [Eurobench context](https://github.com/eurobench/docker_test).
This requires `python3`.

```shell
# from the root repository
# download the generic test file
wget -O test_docker_call.py https://raw.githubusercontent.com/eurobench/docker_test/master/test_docker_call.py
# set environment variables according to this repo spec.
export TEST_PLAN=tests/test_plan.xml
export DOCKER_IMAGE=pi_udbenchmark
# launch the script test according to the plan
python3 test_docker_call.py
```

Test plan is defined in file [test_plan.xml](tests/test_plan.xml).

## Acknowledgements

<a href="http://eurobench2020.eu">
  <img src="http://eurobench2020.eu/wp-content/uploads/2018/06/cropped-logoweb.png"
       alt="rosin_logo" height="60" >
</a>

Supported by Eurobench - the European robotic platform for bipedal locomotion benchmarking.
More information: [Eurobench website][eurobench_website]

<img src="http://eurobench2020.eu/wp-content/uploads/2018/02/euflag.png"
     alt="eu_flag" width="100" align="left" >

This project has received funding from the European Union’s Horizon 2020
research and innovation programme under grant agreement no. 779963.

The opinions and arguments expressed reflect only the author‘s view and
reflect in no way the European Commission‘s opinions.
The European Commission is not responsible for any use that may be made
of the information it contains.

[eurobench_logo]: http://eurobench2020.eu/wp-content/uploads/2018/06/cropped-logoweb.png
[eurobench_website]: http://eurobench2020.eu "Go to website"
