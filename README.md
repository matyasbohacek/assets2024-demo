
![example-interaction](https://github.com/user-attachments/assets/638656a5-3091-48bc-b813-fe63ecbf57aa)

# Pose-aware Large Language Model Interface for Providing Feedback to Sign Language Learners

#### [Vaclav Knapp](https://www.linkedin.com/in/václav-knapp-7696b624a/) and [Matyas Bohacek](https://www.matyasbohacek.com)

Sign language learners often find it challenging to self-identify and correct mistakes, and so many turn to automated methods that provide sign language feedback. However, they find that existing methods either require specialized equipment or lack robustness. They, therefore, have to seek human tutors or give up on the inquiry altogether. To overcome the barriers in accessibility and robustness, we build a large language model (LLM)-based tool for that provide feedback to sign language learners. The tool can analyze videos from diverse camera and background settings without specialized equipment thanks to a sign language segmentation and keyframe identification model. Using a pose-aware LLM, the tool can then produce feedback in written language. We present our tool as a demo web application, opening its implementation into specialized learning applications.

> [See paper](https://dl.acm.org/doi/10.1145/3663548.3688515) — [See poster]() — [Contact us](mailto:maty-at-stanford-dot-edu)
> 
> _Published at the ACM ASSETS 2024 conference_

## Getting Started

1. Clone this repo:

```git clone https://github.com/matyasbohacek/assets2024-demo.git```

2. In the `assets2024-demo` directory, set up a Python environment (Python 3.9 is recommended); you can create the environment from scratch or using:

```conda env create -f environment.yml```

3. Install required packages:

```shell
apt-get update 
sudo apt-get install ffmpeg
sudo apt-get install unzip
```

4. If you created your Conda environment from the yaml file, you can skip this step; otherwise, install Python dependencies using:

```pip install -r requirements.txt```

5. In the `assets2024-demo` directory, clone the [PoseGPT repo](https://github.com/yfeng95/PoseGP) using:

```git clone https://github.com/yfeng95/PoseGPT```

6. In the `PoseGPT` directory, run:

```cd PoseGPT && bash fetch_data.sh```

7. Back in the `assets2024-demo` directory, clone the [MS-TCN sign segmentation model](https://github.com/RenzKa/sign-segmentation) using:

```git clone https://github.com/RenzKa/sign-segmentation```

8. Move neceassary files:

```shell
mv sign-segmentation/demo/models PoseGPT/
mv PoseGPT/models PoseGPT/sign_utils
mv sign-segmentation/demo/utils_demo.py PoseGPT/
mv PoseGPT/utils_demo.py PoseGPT/signseg_utils.py
mv tmp PoseGPT/
```

6. This step may not be necessary on your machine, but we found that some machines require additional setup of the gcc and g++ versions:

```shell
find ./assets/PoseGPT/ -type f -exec sed -i 's/gcc-7/gcc-11/g' {} +
find ./assets/PoseGPT/ -type f -exec sed -i 's/g++-7/g++-11/g' {} +
```

7. Finally, install necessary models from the `sign-segmentation` repo and place them in `./sign-segmentation/models`.

## Customization

**Key Frame Identification.** Our pipeline identifies key frames in the student and reference sign videos. This can be done using the MS-TCN sign segmentation backbone (*Renz et al.*). While this usually works, we found that this solution sometimes leads to incorrect results, especially with unseen signs. An alternative solution can be to use the middle frame of each video, which we found to be effective in most cases.

**LLM Backbone.** Our implementation utilizes the PoseGPT LLM (*Feng et al.*). However, it could be replaced by any multimodal LLM that can process text prompts and images and effectively comment on poses.

**Reference Selection.** We provide three reference example signs: hello, phone, and think. You can modify these by adding reference frames in the `PoseGPT/tmp` folder and marking these signs in the `working_app.py` dictionary on lines ``.

## Citation

```bibtex
@inproceedings{10.1145/3663548.3688515,
    author = {Knapp, Vaclav and Bohacek, Matyas},
    title = {Pose-aware Large Language Model Interface for Providing Feedback to Sign Language Learners},
    year = {2024},
    isbn = {9798400706776},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3663548.3688515},
    doi = {10.1145/3663548.3688515},
    booktitle = {Proceedings of the 26th International ACM SIGACCESS Conference on Computers and Accessibility},
    articleno = {121},
    numpages = {5},
    keywords = {Large Language Models, Learning Tool, Sign Language},
    location = {St. John's, NL, Canada},
    series = {ASSETS '24}
}
```

## Remarks & Updates

- (**October 28, 2024**) The work is presented at ASSETS 2024 as a demo!
