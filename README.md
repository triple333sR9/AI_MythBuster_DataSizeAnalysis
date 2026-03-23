## Investigating whether more data is always beneficial when training machine learning models

### 1. Introduction
This report uses an over simplistic concept of an artificial intelligence model and data usage to demostrate the effects 
of increased data set size on the model's performance. A simple search of how much data exists currently on the world wide web 
results in varying values and estimates, depending on wording of the question one might see ranges from over a hundread zettabytes 
to defining petabytes, all without references for the origin of their numbers. One such claim comes from BBC, claiming that 
"the big four store at least 1,200 petabytes between them" (Mitchell, https://www.sciencefocus.com/future-technology/how-much-data-is-on-the-internet). 
Regardless, the illustration of the situation is clear, there is more data both in existence and being generated every second 
than what puny little human minds can imagine. Yet much emphasis is placed on creating more data originating from the idea 
that the industry is running out of data to train AI models on (https://theconversation.com/researchers-warn-we-could-run-out-of-data-to-train-ai-by-2026-what-then-216741). 
Whilst this particular article is dated and did not age too well for this report, written in 2026, Big AI is still training new models without 
the sky falling down. 

Whilst it is unrealistic to investigate whether there exists too much data, this study proposes an alternative. Using a 10-Class
CNN classification model and a small set of commonly used data set in the machine learning community. The goal is to challenge the 
assumptions conceived regarding the amount of data needed for a _good_ model.

### 2. Methodology
To emphasise the effect of increased data, all other variables are set (see config.py). This includes the
architecture of the model which is a 3-layer CNN, a relatively generic model for image detection. Other controlled
variables include learning rate and weight decay rate, number of epoch, batch size, early stopping patience, drop out
level, seeds and data set.

| Hyperparameter          | Value                                                                                                                                                                          |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dropout                 | 0.5                                                                                                                                                                            |
| epochs                  | 50                                                                                                                                                                             |
| early stopping patience | 10                                                                                                                                                                             |
| learning rate           | 0.001                                                                                                                                                                          |
| weight decay            | 0.0001                                                                                                                                                                         |
| seeds                   | [14994796, 20194682, 31380194, 73962792, 99719928]                                                                                                                             |
| CNN layer 1             | input: 1 channel (grayscale), Output: 32 channels, 3x3 kernel with padding=1, batch normalization=32, ReLU activation, 2x2 max pooling (reduces spatial dimensions by half)    |
| CNN layer 2             | input: 32 channel (grayscale), Output: 64 channels, 3x3 kernel with padding=1, batch normalization=64, ReLU activation, 2x2 max pooling (reduces spatial dimensions by half)   |
| CNN layer 3             | input: 64 channel (grayscale), Output: 128 channels, 3x3 kernel with padding=1, batch normalization=128, ReLU activation, 2x2 max pooling (reduces spatial dimensions by half) |
| Classifier layer        | default flatten, linear 128*3*3 input features to 256 output features, ReLU activation, dropout of 0.5, linear 256 features into number of classes                             |
| Validation split        | 0.1 of training data set                                                                                                                                                       |
| Training data set       | 60 000 images total                                                                                                                                                            |
| Test data set           | 10 000 images total                                                                                                                                                            |
| Data set                | Fashion-MNIST                                                                                                                                                                               |

Table ?: List of hyperparameters fixed for all experiments.

Each seed is considered a unique run of the experiment. Effectively, each condition is tested five times and the averaged
metrics of all five are used in the conclusion. See Appendix for raw data.

#### 2.1 Increasing amount of clean training data for a 10-class CNN
The experiment randomly samples the training data using 5% to 100% of each class's training data to train the model to
maintain class distribution. All ten classes are used and models are to predict all ten classes.

The hypothesis is that increasing the amount of data will have limited effects on model performance beyond a certain threshold, thus the point
of diminishing returns. Unlike the intuition of the scaling law which typically predicts a linear proportional relationship, 
a model can only learn so much from the data it has due to the limits of current architecture infrastructure and thus there
exists a point whereby a model can consume too much data with little improvement. The performance metrics should show a
relatively substantial increase initially from the point of not enough data to an optimal point, then the increase in performance
metrics slow drastically and perhaps even decrease due to overfitting.

The null hypothesis is that the change in performance metrics does not substantially slow down as the data increases beyond a certain threshold.

| Percentage (%) | Training data amount | Validation data amount |
|----------------|----------------------|------------------------|
| 5              | 2 700                | 300                    |
| 10             | 5 400                | 600                    |
| 15             | 9 000                | 900                    |
| 20             | 11 800               | 1 200                  |
| 25             | 14 500               | 1 500                  |
| 30             | 16 200               | 1 800                  |
| 35             | 18 900               | 2 100                  |
| 40             | 21 600               | 2 400                  |
| 45             | 24 300               | 2 700                  |
| 50             | 27 000               | 3 000                  |
| 55             | 29 700               | 3 300                  |
| 60             | 32 400               | 3 600                  |
| 65             | 35 100               | 3 900                  |
| 70             | 37 800               | 4 200                  |
| 75             | 40 500               | 4 500                  |
| 80             | 43 200               | 4 800                  |
| 85             | 45 900               | 5 100                  |
| 90             | 48 600               | 5 400                  |
| 95             | 51 300               | 5 700                  |
| 100            | 54 000               | 6 000                  |
Table ?: Number of images used for training and validation per pecentage.

The experiment 1 script systematically samples data randomly whilst keeping the distribution of the classes, using a seed. 
For each data size condition, a model is trained and evaluated. The result looks at the aggregation of all metrics across all models and data size conditions.

#### 2.2 Increased training data contains varying amounts of noise for a 10-class CNN
Models are typically considered in the context of a perfect world, with clean and clear data sets. Building on top of 
the first experiment, this experiment explores the effect of noisy data's ratio to clean data on model performance.

The hypothesis is that the cleaner the data, the better the performance. Noisier data will cause model performance to decrease.

The null hypothesis is that adding noisy data would not change the performance of the models significantly and the 
results would look similar to that of experiment 1.

The 0% noisy data results came from experiment 1 to save on time and resources.

| Data set parameter                                   | Value ranges                                     |
|------------------------------------------------------|--------------------------------------------------|
| Number of clean data                                 | 30 000 (fixed)                                   |
| Probability that a label is flipped (noise rate (%)) | [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] |
| Rate of noisy data to clean data  (%)                | [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]       |
Table (?): Parameters specific to experiment 2 (see config.py).

| Noisy Percentage (%) | Added Noisy Data | Noise Rate (%) | Data with flipped label (estimated) | Total data |
|----------------------|------------------|----------------|-------------------------------------|------------| 
| 5                    | 3 000            | 5              | 150                                 | 33 000     |
| 5                    | 3 000            | 10             | 300                                 | 33 000     |
| 5                    | 3 000            | 15             | 450                                 | 33 000     |
| 5                    | 3 000            | 20             | 600                                 | 33 000     |
| 5                    | 3 000            | 25             | 750                                 | 33 000     |
| 5                    | 3 000            | 30             | 900                                 | 33 000     |
| 5                    | 3 000            | 35             | 1 200                               | 33 000     |
| 5                    | 3 000            | 40             | 1 500                               | 33 000     |
| 10                   | 6 000            | 5              | 300                                 | 36 000     |
| 10                   | 6 000            | 10             | 600                                 | 36 000     |
| 10                   | 6 000            | 15             | 900                                 | 36 000     |
| 10                   | 6 000            | 20             | 1 200                               | 36 000     |
| 10                   | 6 000            | 25             | 1 500                               | 36 000     |
| 10                   | 6 000            | 30             | 1 800                               | 36 000     |
| 10                   | 6 000            | 35             | 2 100                               | 36 000     |
| 10                   | 6 000            | 40             | 2 400                               | 36 000     |
| 15                   | 9 000            | 5              | 450                                 | 39 000     |
| 15                   | 9 000            | 10             | 900                                 | 39 000     |
| 15                   | 9 000            | 15             | 1 350                               | 39 000     |
| 15                   | 9 000            | 20             | 1 800                               | 39 000     |
| 15                   | 9 000            | 25             | 2 250                               | 39 000     |
| 15                   | 9 000            | 30             | 2 700                               | 39 000     |
| 15                   | 9 000            | 35             | 3 150                               | 39 000     |
| 15                   | 9 000            | 40             | 3 600                               | 39 000     |
| 20                   | 12 000           | 5              | 600                                 | 42 000     |
| 20                   | 12 000           | 10             | 1 200                               | 42 000     |
| 20                   | 12 000           | 15             | 1 800                               | 42 000     |
| 20                   | 12 000           | 20             | 2 400                               | 42 000     |
| 20                   | 12 000           | 25             | 3 000                               | 42 000     |
| 20                   | 12 000           | 30             | 3 600                               | 42 000     |
| 20                   | 12 000           | 35             | 4 200                               | 42 000     |
| 20                   | 12 000           | 40             | 4 800                               | 42 000     |
| 25                   | 15 000           | 5              | 750                                 | 45 000     |
| 25                   | 15 000           | 10             | 1 500                               | 45 000     |
| 25                   | 15 000           | 15             | 2 250                               | 45 000     |
| 25                   | 15 000           | 20             | 3 000                               | 45 000     |
| 25                   | 15 000           | 25             | 3 750                               | 45 000     |
| 25                   | 15 000           | 30             | 4 500                               | 45 000     |
| 25                   | 15 000           | 35             | 5 250                               | 45 000     |
| 25                   | 15 000           | 40             | 6 000                               | 45 000     |
| 30                   | 18 000           | 5              | 900                                 | 48 000     |
| 30                   | 18 000           | 10             | 1 800                               | 48 000     |
| 30                   | 18 000           | 15             | 2 700                               | 48 000     |
| 30                   | 18 000           | 20             | 3 600                               | 48 000     |
| 30                   | 18 000           | 25             | 4 500                               | 48 000     |
| 30                   | 18 000           | 30             | 5 400                               | 48 000     |
| 30                   | 18 000           | 35             | 6 300                               | 48 000     |
| 30                   | 18 000           | 40             | 7 200                               | 48 000     |
| 35                   | 21 000           | 5              | 1 050                               | 51 000     |
| 35                   | 21 000           | 10             | 2 100                               | 51 000     |
| 35                   | 21 000           | 15             | 3 150                               | 51 000     |
| 35                   | 21 000           | 20             | 4 200                               | 51 000     |
| 35                   | 21 000           | 25             | 5 250                               | 51 000     |
| 35                   | 21 000           | 30             | 6 300                               | 51 000     |
| 35                   | 21 000           | 35             | 7 350                               | 51 000     |
| 35                   | 21 000           | 40             | 8 400                               | 51 000     |
| 40                   | 24 000           | 5              | 1 200                               | 54 000     |
| 40                   | 24 000           | 10             | 2 400                               | 54 000     |
| 40                   | 24 000           | 15             | 3 600                               | 54 000     |
| 40                   | 24 000           | 20             | 4 800                               | 54 000     |
| 40                   | 24 000           | 25             | 6 000                               | 54 000     |
| 40                   | 24 000           | 30             | 7 200                               | 54 000     |
| 40                   | 24 000           | 35             | 8 400                               | 54 000     |
| 40                   | 24 000           | 40             | 9 600                               | 54 000     |
| 45                   | 27 000           | 5              | 1 350                               | 57 000     |
| 45                   | 27 000           | 10             | 2 700                               | 57 000     |
| 45                   | 27 000           | 15             | 4 050                               | 57 000     |
| 45                   | 27 000           | 20             | 5 400                               | 57 000     |
| 45                   | 27 000           | 25             | 6 750                               | 57 000     |
| 45                   | 27 000           | 30             | 8 100                               | 57 000     |
| 45                   | 27 000           | 35             | 9 450                               | 57 000     |
| 45                   | 27 000           | 40             | 10 800                              | 57 000     |
| 50                   | 30 000           | 5              | 1 500                               | 60 000     |
| 50                   | 30 000           | 10             | 3 000                               | 60 000     |
| 50                   | 30 000           | 15             | 4 500                               | 60 000     |
| 50                   | 30 000           | 20             | 6 000                               | 60 000     |
| 50                   | 30 000           | 25             | 7 500                               | 60 000     |
| 50                   | 30 000           | 30             | 9 000                               | 60 000     |
| 50                   | 30 000           | 35             | 10 500                              | 60 000     |
| 50                   | 30 000           | 40             | 12 000                              | 60 000     |
Table (?): Expanded list of combinations of noise percentage to noise rates and absolute value of training data used

Using the same seeds as experiment, the data is randomly sampled for the clean data set and for the noisy. A noisy data has 
is clean data where every data point has a chance of having its labels flipped. This chance is determined by the noise rate. 
For every row condition in table (?) above, a model was trained for each seed (total of 5) and evaluated. The results are
an aggregate analysis of all models' metrics.

#### 2.3 Increased training data and its effects on differing numbers of classes for a classification CNN
The results of experiment 1 and experiment 2 opens the question to whether the effects are the same with varying numbers of classes.
To further control this experiment, only clean data was used to emphasise the effect of number of classes.

The hypothesis is that fewer classes to be classified by the model, the less data required. Thus, the performance metrics should show 
better performance

For each number of classes, the labels are sample randomly for five combinations to account for the variation in the object itself
and the ease of differentiating for different classes. This allows the possibility for classes that are easily differentiated to be 
paired together with the same likelihood as classes that are difficult to differentiate, making the groupings effectively arbitrary.

| Number of Classes | Percentage of total data per class | Total data set size |
|-------------------|------------------------------------|---------------------|
| 2                 | 10                                 | 600                 |
| 2                 | 20                                 | 1 200               |
| 2                 | 30                                 | 1 800               |
| 2                 | 40                                 | 2 400               |
| 2                 | 50                                 | 3 000               |
| 2                 | 60                                 | 3 600               |
| 2                 | 70                                 | 4 200               |
| 2                 | 80                                 | 4 800               |
| 2                 | 90                                 | 5 400               |
| 2                 | 100                                | 6 000               |
| 3                 | 10                                 | 900                 |
| 3                 | 20                                 | 1 800               |
| 3                 | 30                                 | 2 700               |
| 3                 | 40                                 | 3 600               |
| 3                 | 50                                 | 4 500               |
| 3                 | 60                                 | 5 400               |
| 3                 | 70                                 | 6 300               |
| 3                 | 80                                 | 7 200               |
| 3                 | 90                                 | 8 100               |
| 3                 | 100                                | 9 000               |
| 4                 | 10                                 | 1 200               |
| 4                 | 20                                 | 2 400               |
| 4                 | 30                                 | 3 600               |
| 4                 | 40                                 | 4 800               |
| 4                 | 50                                 | 6 000               |
| 4                 | 60                                 | 7 200               |
| 4                 | 70                                 | 8 400               |
| 4                 | 80                                 | 9 600               |
| 4                 | 90                                 | 10 800              |
| 4                 | 100                                | 12 000              |
| 5                 | 10                                 | 1 500               |
| 5                 | 20                                 | 3 000               |
| 5                 | 30                                 | 4 500               |
| 5                 | 40                                 | 6 000               |
| 5                 | 50                                 | 7 500               |
| 5                 | 60                                 | 9 000               |
| 5                 | 70                                 | 10 500              |
| 5                 | 80                                 | 12 000              |
| 5                 | 90                                 | 13 500              |
| 5                 | 100                                | 15 000              |
| 6                 | 10                                 | 1 800               |
| 6                 | 20                                 | 3 600               |
| 6                 | 30                                 | 5 400               |
| 6                 | 40                                 | 7 200               |
| 6                 | 50                                 | 9 000               |
| 6                 | 60                                 | 10 800              |
| 6                 | 70                                 | 12 600              |
| 6                 | 80                                 | 14 400              |
| 6                 | 90                                 | 16 200              |
| 6                 | 100                                | 18 000              |
| 7                 | 10                                 | 2 100               |
| 7                 | 20                                 | 4 200               |
| 7                 | 30                                 | 6 300               |
| 7                 | 40                                 | 8 400               |
| 7                 | 50                                 | 10 500              |
| 7                 | 60                                 | 12 600              |
| 7                 | 70                                 | 14 700              |
| 7                 | 80                                 | 16 800              |
| 7                 | 90                                 | 18 900              |
| 7                 | 100                                | 21 000              |
| 8                 | 10                                 | 2 400               |
| 8                 | 20                                 | 4 800               |
| 8                 | 30                                 | 7 200               |
| 8                 | 40                                 | 9 600               |
| 8                 | 50                                 | 12 000              |
| 8                 | 60                                 | 14 400              |
| 8                 | 70                                 | 16 800              |
| 8                 | 80                                 | 19 200              |
| 8                 | 90                                 | 21 600              |
| 8                 | 100                                | 24 000              |
| 9                 | 10                                 | 2 700               |
| 9                 | 20                                 | 5 400               |
| 9                 | 30                                 | 8 100               |
| 9                 | 40                                 | 10 800              |
| 9                 | 50                                 | 13 500              |
| 9                 | 60                                 | 16 200              |
| 9                 | 70                                 | 18 900              |
| 9                 | 80                                 | 21 600              |
| 9                 | 90                                 | 24 300              |
| 9                 | 100                                | 27 000              |
| 10                | 10                                 | 3 000               |
| 10                | 20                                 | 6 000               |
| 10                | 30                                 | 9 000               |
| 10                | 40                                 | 12 000              |
| 10                | 50                                 | 15 000              |
| 10                | 60                                 | 18 000              |
| 10                | 70                                 | 21 000              |
| 10                | 80                                 | 24 000              |
| 10                | 90                                 | 27 000              |
| 10                | 100                                | 30 000              |
Table (?): 

### 3. Results
See results folder for all graphs and JSONs files generated.

The conjugate results look at where on the graph is f(x) maximised, meaning where the data most efficiently produces the 
best results. The formulas used are as follows.

##### 3.0.1 Efficiency x and Efficiency y
Whereby metric = f(x) is given by the formula of the line of best fit, is maximised.

Efficiency_x = f(x)/x
Efficiency_y = f(efficiency_x)

The optimal point (maximised point) is discovered by taking the derivative of f(x)/x and setting it to zero and solve for x.

###### Logarithmic: $f(x) = a \ln(x) + b$

Optimization:
$$\text{Maximize: } \frac{f(x)}{x} = \frac{a \ln(x) + b}{x}$$

Take derivative:
$$\frac{d}{dx}\left[\frac{a \ln(x) + b}{x}\right] = \frac{a/x \cdot x - (a \ln(x) + b) \cdot 1}{x^2} = \frac{a - a\ln(x) - b}{x^2}$$

Set equal to zero:
$$a - a\ln(x) - b = 0$$
$$a\ln(x) = a - b$$
$$\ln(x) = 1 - \frac{b}{a}$$

**Solution**:
$$x_{\text{opt}} = e^{1 - b/a}$$
$$y_{\text{opt}} = a \ln(x_{\text{opt}}) + b$$

###### Linear: $f(x) = ax + b$

$$\frac{f(x)}{x} = a + \frac{b}{x}$$

- Maximized at $x_{\min}$ if $b \geq 0$
- Maximized at $x_{\max}$ if $b < 0$


##### 3.0.2 Knee/Elbow Point
The knee point is defined as the point on the curve that is furthest away from the diagonal line $y=x$ in normalised space. 
The mathematical intuition is that at small $x$, $y$ grows rapidly and due to its closeness to origin, the point does not end 
up very far from the $y=x$ line. When the growth of $y$ finally plateaus, the graph will converge towards (1, 1). So the point at which 
$y$ is furtherest from shows an optimal efficiency, hence the knee.

###### Normalization
 
Transform x and y to [0, 1] range:
 
$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
 
$$y_{\text{norm}} = \frac{y - y_{\min}}{y_{\max} - y_{\min}}$$
 
###### Distance Calculation
 
Distance from the diagonal line $y = x$:
 
$$d(x) = |y_{\text{norm}}(x) - x_{\text{norm}}(x)|$$
 
###### Knee Point
 
$$x_{\text{knee}} = \arg\max_{x \in [x_{\min}, x_{\max}]} |y_{\text{norm}}(x) - x_{\text{norm}}(x)|$$
 
$$y_{\text{knee}} = f(x_{\text{knee}})$$
 
**Note**: No knee exists for linear curves


##### 3.0.3 Marginal Threshold Point
The marginal threshold looks at the marginal gain. Specifically, the threshold is the point where each additional 
sample only improves accuracy by ~0.0001 (default threshold) or less.

###### Logarithmic Curve: $f(x) = a \ln(x) + b$
 
**Derivative**:
$$\frac{dy}{dx} = \frac{a}{x}$$
 
**Set Equal to Threshold**:
$$\frac{a}{x} = \tau$$
 
**Solution**:
$$x_{\text{marginal}} = \frac{a}{\tau}$$
 
$$y_{\text{marginal}} = a \ln\left(\frac{a}{\tau}\right) + b$$
 
###### Linear Curve: $f(x) = ax + b$
 
**Derivative**:
$$\frac{dy}{dx} = a \quad \text{(constant)}$$
 
**Solution**:
- If $|a| \leq \tau$: threshold already met at $x_{\min}$ (returns `"constant"`)
- If $|a| > \tau$: threshold never met (returns `"above_thresh"`)
 
###### Power Curve: $f(x) = ax^c + b$
 
**Derivative**:
$$\frac{dy}{dx} = acx^{c-1}$$
 
**Set Equal to Threshold**:
$$acx^{c-1} = \tau$$
 
$$x^{c-1} = \frac{\tau}{ac}$$
 
**Solution** (if $c \neq 1$ and $ac > 0$):
$$x_{\text{marginal}} = \left(\frac{\tau}{ac}\right)^{\frac{1}{c-1}}$$
 
$$y_{\text{marginal}} = a\left(x_{\text{marginal}}\right)^c + b$$


#### 3.1 Increasing amount of clean training data for a 10-class CNN Results
The most obvious result is that after a certain point, the increase gained by the increment of added 
data alone no longer sufficiently improves the model's performance. All results for this experiment fcan be found under $/results/exp1$ folder. 

![accuracy_absolute.png](code/results/exp1/figures/accuracy_absolute.png)
Figure (?): The effect of training data set size on model accuracy

![accuracy_pct.png](code/results/exp1/figures/accuracy_pct.png)
Figure (?): The effect of training data set size as a percentage of full data set available on model accuracy

Both figures portraying the accuracy of the model through this experiment showed a diminishing returns effect on accuracy displayed by the 
model as the amount of data provided for the training process increased. 

| func_type | equation               | r_squared | x_min | x_max  | efficiency_x | efficiency_y | efficiency_note | efficiency_ratio | knee_x   | knee_y | knee_note    | marginal_x | marginal_y | marginal_note | marginal_threshold | metric            | 
|-----------|------------------------|-----------|-------|--------|--------------|--------------|-----------------|------------------|----------|--------|--------------|------------|------------|---------------|--------------------|-------------------|
| log       | 0.0218·log(x) + 0.6820 | 0.986     | 3 000 | 60 000 | 3 001        | 0.8563       | analytical        | 0.00028543       | 19033.03 | 0.8965 | max_distance | 3000       | 0.8563     | dy/dx=0.0001  | 0.0001             | accuracy          |
| log       | 0.0218·log(x) + 0.6820 | 0.986     | 3 000 | 60 000 | 3 001        | 0.8563       | analytical        | 0.00028543       | 19033.03 | 0.8965 | max_distance | 3000       | 0.8563     | dy/dx=0.0001  | 0.0001             | balanced_accuracy |
| log       | 0.0220·log(x) + 0.6796 | 0.9836    | 3 000 | 60 000 | 3 001        | 0.8555       | analytical        | 0.00028516       | 19033.03 | 0.8961 | max_distance | 3000       | 0.8555     | dy/dx=0.0001  | 0.0001             | f1_macro          |
| log       | 0.0027·log(x) + 0.9658 | 0.974     | 3 000 | 60 000 | 3 001        | 0.9875       | analytical        | 0.00032918       | 19033.03 | 0.9925 | max_distance | 3000       | 0.9875     | dy/dx=0.0001  | 0.0001             | auroc_macro       |
| log       | 0.0167·log(x) + 0.7862 | 0.9806    | 3 000 | 60 000 | 3 001        | 0.9197       | analytical        | 0.00030657       | 19033.03 | 0.9505 | max_distance | 3000       | 0.9197     | dy/dx=0.0001  | 0.0001             | auprc_macro       |
| log       | 0.0240·log(x) + 0.6487 | 0.9868    | 3 000 | 60 000 | 3 001        | 0.8409       | analytical        | 0.00028031       | 19033.03 | 0.8853 | max_distance | 3000       | 0.8409     | dy/dx=0.0001  | 0.0001             | mcc               |
| log       | 0.0207·log(x) + 0.6942 | 0.9888    | 3 000 | 60 000 | 3 001        | 0.8598       | analytical        | 0.0002866        | 19033.03 | 0.8980 | max_distance | 3000       | 0.8598     | dy/dx=0.0001  | 0.0001             | precision_macro   |
| log       | 0.0218·log(x) + 0.6820 | 0.986     | 3 000 | 60 000 | 3 001        | 0.8563       | analytical        | 0.00028543       | 19033.03 | 0.8965 | max_distance | 3000       | 0.8563     | dy/dx=0.0001  | 0.0001             | recall_macro      |
Table (?): 

The efficiency for all performance metrics falls at the same $x$ value of 3001 data sampless. Whilst the y-value varies, it substantially 
illustrates the point that optimal efficiency of the performance metrics is low. The formula used in the code for this is normalised for the line of best fit.

The point for the knee shows the most divergent numbers from the other two metrics as it does not sit on the $x$ axis boundary for this experiment, 
instead it is at $x=19,033$. This is the point that the performance metrics improvement begin to plateau to suboptimal levels.

Marginal threshold values here are unhelpful as the line of best fit places the point at which each additional 
sample only improves accuracy by ~0.0001 (default threshold) or less at a value smaller than the minimal dataset size tested. 
Whilst this provides a good idea that even at 3000 data points, each additional sample's improvement to the model may be suboptimal.

Interestingly, the three different measures of optimal point suggests different values at which the usage of data provides the greatest improvements 
to the model's performance. This ranges from a value well below the initialising data set size (for efficiency), to being on the boundary (for marginal gain 
threshold) to a value roughly midway through the tested data set (at 19,033 or 63.4%). None of the values come anywhere close to the size of the full data set provided 
at 30, 000. This illustrates the assumption from the perspective of data set curators that perhaps too much data was provided on the assumption that a lot of data is needed. 
For classification models, the presumed amount of data required may be much higher in the minds of humans than what the model 
actually calls for. 

#### 3.2 Increased training data contains varying amounts of noise for a 10-class CNN Results

#### 3.3 Increasing amount of clean training data for a 10-class CNN Results

### 4. Discussion and Interpretation
...

### 5. Conclusion
...


### 6. Appendix

https://link.springer.com/article/10.1007/s11263-015-0812-2
https://www.tandfonline.com/doi/abs/10.1080/01431169508954507
https://proceedings.mlr.press/r1/oates97b.html
https://www.sciencedirect.com/science/article/pii/S0341816216302090
https://www.sciencedirect.com/science/article/pii/S0895435618310813
https://www.sciencedirect.com/science/article/pii/S0034425705002750
https://link.springer.com/article/10.1186/bcr2468
https://www.mdpi.com/2072-4292/13/3/368
https://dl.acm.org/doi/abs/10.1145/1390156.1390273
https://www.sciencedirect.com/science/article/pii/S0165993606002330
https://www.sciencedirect.com/science/article/pii/S1574954120300352
https://www.sciencedirect.com/science/article/pii/S0034425706001234
https://www.mdpi.com/1420-3049/26/4/1111
https://elar.khmnu.edu.ua/items/a5d0d900-449d-4f3d-b800-c7200084384f
https://ufal.mff.cuni.cz/books/preview/2018-zeman_full.pdf
https://aclanthology.org/D14-1096.pdf
https://www.sciencedirect.com/science/article/pii/S1470160X14002088
https://esajournals.onlinelibrary.wiley.com/doi/epdf/10.1002/ecs2.70205

how many tokens in chatgpt's model:
https://explodingtopics.com/blog/gpt-parameters 

How much data exists in the world:
https://www.sciencefocus.com/future-technology/how-much-data-is-on-the-internet