# ASSETS 2024 Demo

<br>

> ### Pose-aware Large Language Model Interface for Providing Feedback to Sign Language Learners
> Sign language learners often find it challenging to self-identify and correct mistakes, and so many turn to automated methods providing sign language feedback. However, they find that existing methods either require specialized equipment or lack robustness. They, therefore, have to seek human tutors or resign on the inquiry altogether. To overcome the barriers in accessibility and robustness, we build a large language model (LLM)-based tool for providing feedback to sign language learners. The tool can analyze videos from diverse camera and background settings without specialized equipment thanks to a sign language segmentation and keyframe identification model. Using a pose-aware LLM, the tool can then produce feedback in natural language. We present our tool as a demo web application, opening its implementation into specialized learning applications.

<br>

## TODOs:

### Technical

#### High-priority

- [ ] (Vasek) ChatPose wrapper
- [ ] (Vasek) Chatting demo interface
- [ ] (Vasek) show selected key frames in the UI

### Mid-priority

- [ ] (Maty & Vasek) UI, CSS
- [ ] (Maty & Vasek) HuggingFace Demos/Spaces Hosting

#### Low-priority

- [ ] (Vasek) add more probes and non-manual components to the prompt
- [ ] (Maty & Vasek) instructions, FAQ

### Paper Writing

- [ ] (Maty) identify questions/discussions to be held at the conference and incorporate them into the conclusion (reviewer #1)
- [ ] (Maty) change of wording "natural language" -> "written language" (reviewer #2)
- [ ] (Maty) mention that the feedback may be delivered directly in the form of sign language in the future (reviewer #2)
- [ ] (Maty) add more probes and non-manual components to the prompt; mention this in the paper (reviewer #2)
- [ ] (Maty) show selected key frames in the UI; reflect this in the figures (screenshots); mention this in the paper (reviewer #3)
- [ ] (Maty) clarify pipeline details (reviewer #3)
- [ ] (Maty) clarify how prompts (questions) were chosen and that there is room for improvement in this regard (reviewer #3)
- [ ] (Maty) example results, both positive and negative (reviewer #3)
