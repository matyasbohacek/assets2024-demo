
![example-interaction](https://github.com/user-attachments/assets/638656a5-3091-48bc-b813-fe63ecbf57aa)

# Pose-aware Large Language Model Interface for Providing Feedback to Sign Language Learners

#### [Vaclav Knapp](https://www.linkedin.com/in/václav-knapp-7696b624a/) and [Matyas Bohacek](https://www.matyasbohacek.com)

Sign language learners often find it challenging to self-identify and correct mistakes, and so many turn to automated methods that provide sign language feedback. However, they find that existing methods either require specialized equipment or lack robustness. They, therefore, have to seek human tutors or give up on the inquiry altogether. To overcome the barriers in accessibility and robustness, we build a large language model (LLM)-based tool for that provide feedback to sign language learners. The tool can analyze videos from diverse camera and background settings without specialized equipment thanks to a sign language segmentation and keyframe identification model. Using a pose-aware LLM, the tool can then produce feedback in written language. We present our tool as a demo web application, opening its implementation into specialized learning applications.

> [See paper](https://dl.acm.org/doi/10.1145/3663548.3688515) — [See poster]() — [Contact us](mailto:maty-at-stanford-dot-edu)
> 
> _Published at the ACM ASSETS 2024 conference_

## Getting Started

1. Set up a Python environment (Python 3.9 is recommended) and install the dependencies using `pip install -r requirements.txt`. Alternatively, create the environment using `conda env create -f environment.yml`.

2. clone PoseGPT repo in `.` using: `git clone https://github.com/yfeng95/PoseGPT`

3. in PoseGPT repo Folder run 'bash fetch_data.sh'

4. Clone Sign segmentation model to the `.`: `git clone https://github.com/RenzKa/sign-segmentation`

5. Move neceassary files: 
```shell
mv sign-segmentation/demo/models PoseGPT/
mv PoseGPT/models PoseGPT/sign_utils
mv sign-segmentation/demo/utils_demo.py PoseGPT/
mv PoseGPT/utils_demo.py PoseGPT/signseg_utils.py
```

6. Run this to change gcc and g++ versions:
```shell
find ./assets/PoseGPT/ -type f -exec sed -i 's/gcc-7/gcc-11/g' {} +
find ./assets/PoseGPT/ -type f -exec sed -i 's/g++-7/g++-11/g' {} +
```

7. Install neccasary libraries:
```shell
apt-get update 
sudo apt-get install ffmpeg
sudo apt-get install unzip
```

8. Install necessary models from sign-segmentation repo `https://github.com/RenzKa/sign-segmentation` and place them in `./sign-segmentation/models`





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
