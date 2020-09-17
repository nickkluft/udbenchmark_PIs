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
run_fpe tests/data/input/subject_04_jointTrajectories_01.csv tests/data/input/subject_04_comTrajectories_01.csv tests/data/input/subject_04_angMomentum_01.csv tests/data/input/subject_04_comItensor_01.csv tests/data/input/subject_04_gaitEvents_01 tests/data/input/subject_04_testbedLabel_01.yaml out_tests
```

```term
run_fpm tests/data/input/subject_04_jointTrajectories_01.csv tests/data/input/subject_04_comTrajectories_01.csv tests/data/input/subject_04_gaitEvents_01 out_tests
```

```term
run_spattemp tests/data/input/subject_04_jointTrajectories_01.csv tests/data/input/subject_04_gaitEvents_01 out_tests
```

```term
run_lde tests/data/input/subject_04_comTrajectories_01.csv tests/data/input/subject_04_gaitEvents_01 out_tests
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
docker run --rm -v $PWD/tests/data/input:/in -v $PWD/out_tests:/out pi_udbenchmark run_fpe /in/subject_04_jointTrajectories_01.csv /in/subject_04_comTrajectories_01.csv /in/subject_04_angMomentum_01.csv /in/subject_04_comItensor_01.csv /in/subject_04_gaitEvents_01 /in/subject_04_testbedLabel_01.yaml /out
```

```shell
docker run --rm -v $PWD/tests/data/input:/in -v $PWD/out_tests:/out pi_udbenchmark run_fpm /in/subject_04_jointTrajectories_01.csv /in/subject_04_comTrajectories_01.csv /in/subject_04_gaitEvents_01 /out
```

```shell
docker run --rm -v $PWD/tests/data/input:/in -v $PWD/out_tests:/out pi_udbenchmark run_spattemp /in/subject_04_jointTrajectories_01.csv /in/subject_04_gaitEvents_01 /out
```

```shell
docker run --rm -v $PWD/tests/data/input:/in -v $PWD/out_tests:/out pi_udbenchmark run_lde /in/subject_04_comTrajectories_01.csv /in/subject_04_gaitEvents_01 /out
```

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
