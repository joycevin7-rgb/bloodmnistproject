# Deep Learning Project Guideline

## 1. Basic Information

In this project, your team will define a real-world problem, find a suitable dataset, compare two deep learning methods against a simple baseline, and report the work as a short conference-style paper. You are free to choose the application field, the dataset, and the methods, as long as you follow the requirements and restrictions in this document.

| Item               | Details                                                                                |
| ------------------ | -------------------------------------------------------------------------------------- |
| **Duration**       | 3 weeks+                                                                               |
| **Team size**      | **2 to 3 students per team**                                                           |
| **Deadline**       | **1 November 2026, 23:59**                                                             |
| **Platform**       | PCL                                                                                    |
| **Deliverables**   | **One report (PDF)** and **one code archive (ZIP)** per team                           |
| **Training limit** | Each training run must finish with a **free Google Colab GPU** or equivalent hardware. |

### Grading Rubric

| Criterion                         | Points  | What earns full points                                                                                                                                                                                                                                                                                |
| --------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Problem and Dataset**        | 15      | The problem is stated clearly, including its input and output, and the report explains why the problem matters. Clearly explain the dataset, its splitting ratio, leakage checks and the data distribution are reported.                                                                              |
| **2. Methods and Implementation** | 30      | Each method differs in a meaningful design choice. The hypothesis is stated before the results. Your own design decision is clearly justified. Each method is implemented correctly.                                                                                                                  |
| **3. Experiment and Analysis**    | 35      | All models are compared fairly, using the same split, training budget, and evaluation code. Each model is trained with at least 3 seeds and reported as mean ± standard deviation. The results are explained, not only listed, and the hypothesis is clearly answered. Result visualization provided. |
| **4. Report Quality**             | 10      | The report follows the required section structure and the IEEE template, and is 4 to 6 pages long. Figures and tables are clear and readable. The report discusses and correctly cites 3 to 5 related works.                                                                                          |
| **5. Code and Reproducibility**   | 10      | The code runs from start to end without errors. Seeds are set at the top of the code.**README.md** and **requirements.txt** (these are guidelines for us to run your code) are complete. The code is well commented.                                                                                  |
| **Total**                         | **100** |                                                                                                                                                                                                                                                                                                       |

---

## 2. Project Steps

### Step 1. Choose the Story

Choose an application field, for example *agriculture, medicine, environment, manufacturing, finance, or education.* Then define one specific problem in that field and explain why solving it is useful and who would benefit.

You may choose any task that can be solved with deep learning, for example *image classification, object detection, image segmentation, text classification, speech recognition, or time-series forecasting.*

**Requirements**

- State the problem in one sentence that names the input and the output. 
- Choose a task that can be completed within the compute limit.

### Step 2. Find and Prepare the Dataset

Find a dataset that matches your problem. You may use any public source, for example *Kaggle, Hugging Face, torchvision, torchaudio, or MedMNIST*. If you are unsure, choose a dataset from the Recommended Datasets table below.

**Requirements**

- The dataset must be publicly available under a license that permits academic use. Cite its source in the report.
- If the dataset provides an official train, validation, and test split, use it without changes.
- If the dataset provides only a train and test split, keep the test split unchanged and take 15% of the training split as the validation split, using a fixed random seed.
- If the dataset provides no split, create a 70% / 15% / 15% train, validation, and test split.

- Create and save your data split (e.g., as a CSV) before you train any model. Do not change it afterwards.
- Check for data leakage: duplicate or near-duplicate samples across splits, and samples from the same subject in more than one split. 

- Report the data distribution in each split.

**Restrictions**

- Do not use the test set for training, model selection, or hyperparameter tuning. Evaluate on the test set only after all decisions are final.
- If the full dataset is too large for the compute limit, take a random subsample with a fixed seed and describe the procedure in the report.

**Recommended Datasets (this only recommendation, you can choose beyond these)**

| Dataset             | Task                           | Field                 | Access                | Official split     |
| ------------------- | ------------------------------ | --------------------- | --------------------- | ------------------ |
| **PathMNIST**       | Image classification           | Colon pathology       | medmnist package      | Train / val / test |
| **DermaMNIST**      | Image classification           | Dermatology           | medmnist package      | Train / val / test |
| **BloodMNIST**      | Image classification           | Blood cell microscopy | medmnist package      | Train / val / test |
| **EuroSAT**         | Image classification           | Satellite land use    | torchvision           | None               |
| **Oxford-IIIT Pet** | Image segmentation             | Animals               | torchvision           | Train / test       |
| **Speech Commands** | Keyword recognition from audio | Speech                | torchaudio            | Train / val / test |
| **IMDB**            | Text sentiment classification  | Movie reviews         | Hugging Face datasets | Train / test       |



### Step 3. Decide the Methods

Your experiments include at least three models: one **baseline** and two **methods** (Method A and Method B). A method is the complete training setup: architecture, loss function, optimizer, learning rate, and other key training settings.

Each method must differ in at least one meaningful design choice, for example the architecture, the loss function, the data augmentation strategy, the use of pretrained weights, or the training strategy. Changing only the random seed or the number of epochs does not count as a different method.

**Requirements**

- Baseline: a simple model suited to your task (for example, a small MLP, CNN, or RNN) whose layers are defined in your own code and trained from scratch. 
- Hypothesis: before running the experiments, write down which method you expect to perform better and why. Example: "*We expect Method B to achieve a higher macro-F1 than Method A because the class-weighted loss gives more weight to rare classes*."
- If you use pretrained models, at least one of Method A and Method B must include a design decision of your own (for example, *a modified architecture, a chosen loss function, a specific augmentation pipeline, or a training strategy*). Justify this decision in the report.
- Record all hyperparameters of every model.

**Restrictions**

- Use PyTorch.
- Pretrained weights must come from a public source such as torchvision, timm, or Hugging Face, and must be cited.

### Step 4. Run the Experiments

Train and evaluate the baseline and the methods under the same conditions so that differences in results come from the methods, not from the setup.

**Requirements**

- Fair comparison: use the same data split, the same evaluation code, and the same training budget (the same number of epochs or the same early stopping rule) for all three models. Preprocessing must be the same unless the difference is part of the method itself.
- Seeds: fix the seeds of Python random, NumPy, and your deep learning framework. Train each model with at least 3 different seeds and report results as mean ± standard deviation.
- Model selection: choose hyperparameters and checkpoints using the validation set only.
- Metrics: report at least two evaluation metrics suited to your task, on both the validation set and the test set, and justify your choice. Examples: *accuracy and macro-F1 for imbalanced classification, mean IoU and Dice score for segmentation, mAP for object detection, MAE and RMSE for forecasting*.
- Learning curves: plot training and validation loss per epoch for each model. One representative seed per model is sufficient.
- Error breakdown: for classification, plot the confusion matrix of your best model on the test set. For other tasks, provide an equivalent breakdown of errors, for example per-class IoU for segmentation or error per forecast horizon for forecasting.
- Failure cases: show at least 3 test samples where your best model fails, with the model output and the ground truth, and give a hypothesis for each failure.
- Record the parameter count and the training time of each model.
- Results table: summarize the main results in a table with the format below. 

**Results Table Format**

| Model        | Params | [Metric 1] | [Metric 2] |
| ------------ | ------ | ---------- | ---------- |
| **Baseline** | count  | mean ± std | mean ± std |
| **Method A** | count  | mean ± std | mean ± std |
| **Method B** | count  | mean ± std | mean ± std |

*Replace [metric 1] and [metric 2] with the metrics you chose. Add columns if you report more metrics.*

**Optional: Ablation Study**

An ablation study measures how much each component of your best method contributes. Remove or change one component at a time (for example, turn off data augmentation or replace the loss function) while keeping everything else fixed, and compare the results with the full method. This step is optional.

### Step 5. Write the Report

Write the report following the structure in Section 3 of this guideline. Use your recorded results directly. Every number in the report must be reproducible with your submitted code.

---

## 3. Report

Write the report as a short conference paper using the IEEE conference template. Use LaTeX; Overleaf is recommended because all team members can edit the same document online.

| Item               | Requirement                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| **Template**       | IEEE conference template (two-column)                                                                   |
| **Length**         | 4 to 6 pages, including figures and tables. References and the appendix do not count toward this limit. |
| **Language**       | English                                                                                                 |
| **Citation style** | IEEE numbered style, for example [1], [2]                                                               |

### Template Access

- Overleaf: open <https://www.overleaf.com/gallery/tagged/ieee-official>, select **IEEE Conference Template**, and click **Open as Template**.

*Delete all guidance text that comes with the template before you submit.*

### Report Sections

Use the following sections in this order.

| Section                              | What to write                                                                                                                                                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Abstract**                         | 150 to 250 words. Summarize the problem, dataset, methods, main quantitative result, and key finding.                                                                                                                    |
| **1. Introduction**                  | Describe the story and why the problem matters. State the problem, your hypothesis, and a short summary of what you did.                                                                                                 |
| **2. Related Work**                  | Discuss 3 to 5 published works on similar problems or methods, and explain how your work relates to them.                                                                                                                |
| **3. Dataset**                       | Describe the source, license, number of samples, input format (for example, image size or text length), output labels or targets, data distribution per split, split procedure, leakage checks, and preprocessing.       |
| **4. Method**                        | Describe the baseline, Method A, Method B, etc.: architecture, loss function, optimizer, and key design decisions with their justification. For the baseline, show the output shape of each layer, from input to output. |
| **5. Experiments and Results**       | Describe the setup (hardware, epochs, batch size, learning rate, seeds). Present the main results table and learning curves, then compare the methods and state whether the results support your hypothesis.             |
| **6. Ablation Study (optional)**     | Present and discuss the ablation results, if you performed an ablation study.                                                                                                                                            |
| **7. Discussion and Error Analysis** | Discuss the error breakdown (e.g. using the confusion matrix) and the most common error types. State the limitations of your work.                                                                                       |
| **8. Conclusion**                    | Summarize the main findings, answer your hypothesis, and propose future improvements.                                                                                                                                    |
| **References**                       | List every source you used, and cite each one in the main text.                                                                                                                                                          |
| **Appendix**                         | 1) Contribution of each team member. 2) Statement of how AI tools were used.                                                                                                                                             |

## 4. Submission

One member submits for the whole team on the PCL platform before **1 November 2026, 23:59**. Replace XX in the file names with your team number.

| File                   | Contents                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **GroupXX_Report.pdf** | The final report, exported from LaTeX.                                                                                    |
| **GroupXX_Code.zip**   | Notebook(s) or Python scripts, requirements.txt, README.md, and the saved split files (do not submit the dataset itself!) |

### Use of AI Tools

You may use AI tools, for example for coding help or language editing. Describe how you used them in Appendix B of the report. Every team member must be able to explain any part of the code and the report.
