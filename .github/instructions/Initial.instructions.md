---
applyTo: '**'
---
PRECOG RECRUITMENT TASKS 2026
Tasks are centered around four central themes 
NLP 
Computer Vision
Graphs
Quant

Applicants must choose any one theme and complete the task listed under that theme.

You are encouraged to draw upon existing literature, blogs, and other credible resources to inform your design decisions. While doing so, you are expected to clearly justify the choices and assumptions you make.
You are not required to follow the architectures or methods suggested in the task description, if any. Feel free to design and experiment with your own approaches, modify existing ideas, combine techniques, or explore entirely new directions.
Importantly, all experiments are valuable, including those that do not yield positive results. Any approach you attempt, successful or not should be documented and discussed in your report or presentation.
We strongly recommend reviewing relevant prior work and appropriately citing the literature wherever it informs your methodology, analysis, or implementation.
For some tasks there are bonus tasks - to demonstrate that you can go the extra mile (which is an important characteristic of being in our group)! We encourage you to try those bonus tasks. 
To overcome compute constraints:
You can use Google Colab, Kaggle for compute intensive jobs.
Take a subset of data if the dataset is huge - ex. 25% / 50% / 75% of train data

You must attempt this task on your own. Please make sure you cite any external resources that you may use. We will also check your submission for plagiarism, including ChatGPT :) 



Submission:
Put all your code/notebooks in a GitHub repository. Maintain a README.md explaining your codebase, the directory structure, commands to run your project, the dependency libraries used and the approach you followed. 
A Python notebook is mandatory and must clearly log all experiments and results. Submissions consisting only of Python scripts without accountable, logged results will not be considered. All outcomes must be transparent, reproducible, and traceable within the notebook.
The link to the GitHub repository will be asked for during the interview.
Presentation / Report: 
Programming Task : Document the process and compile a presentation / report summarizing your findings, methodologies, and  insights gained from the analysis in a presentation/report. Your report must contain your methodology, findings, results etc. On the first page, It must detail exactly what parts of the task you did, and what parts of the task you did not do and why. The report would be one of the main documents that would help us take your application forward for the next stages.

Evaluation:
You will be evaluated based on how you approach the problem, and not so much on the performance measures like accuracy etc. Although, we would expect you to beat random performance. 
How you present your code and findings. You should justify all the choices you make regarding data, model, and hyperparameters that you use. You should also be able to demonstrate theoretical understanding of the approaches used.
Across all the tasks, your experiments will give you some quantitative measures to indicate the efficacy of your approach (Accuracy, Recall, MAE, MSE etc) - but dig deeper to analyze the predictions. Can you come up with any hypothesis for why your approach fails/succeeds? Can this analysis help you improve on your approach? Creativity in this analysis is what we are looking for!! - SURPRISE US!!.
How do you handle and sample from large real-world datasets, and solve tasks with whatever resources you have access to.. 

For any doubts regarding this task please directly email to the address provided for each task, add the following email in cc - george.rahul@research.iiit.ac.in and debangan.mishra@students.iiit.ac.in. If you don't get a reply within 24 hrs feel free to drop a reminder.

Tasks
The CV Task can be found in this tab CV.
The NLP Task can be found in this tab NLP.
The Graphs Task can be found in this tab Graphs.
The Quant Task can be found in this tab Quant.

“If you wait until you’re ready, you’ll be waiting the rest of your life.” - Lemony Snicket




Congrats Congrats Money Money
“In this business it’s easy to confuse luck with brains.” - Jim Simons
Objective
Develop an end-to-end algorithmic trading pipeline for a universe of N anonymized stocks. Your goal is to transform raw price data into a trading system that maximizes risk-adjusted returns.
We are testing your ability to handle three critical stages of quantitative research: Data Engineering, Strategy Formulation, and Simulation. 
General Instructions/Tips:
There are FOUR TOTAL parts to this task.
Feature Engineering & Data Cleaning
Model Training & Strategy Formulation
Backtesting & Performance Analysis
Statistical Arbitrage Overlay
We will expect at least one Python notebook as your deliverable for submission detailing every part of the three stage pipeline with comments and explanations of the approach taken.
You may use Python scripts and multiple notebooks to modularize and clean up your code. We recommend this approach (ideally, three notebooks, supplementary scripts, and organized outputs).
Do make visuals to illustrate your learnings and hypotheses wherever possible. It helps both you and us have a better time understanding it.
Learn math. Nothing about this task is impossible to learn/implement with the knowledge a student from IIIT has acquired up until their 2nd year, first sem. It may seem daunting, but it’s easier than it looks.
This task is open-ended. Not everything will have a defined, objectively correct end goal. This is reflective of real quant research as well. Feel free to explore well beyond the scope of just this task’s specifications.
KEEP LOOKING BACK AT THIS PAGE FOR POSSIBLE UPDATES
Data
File: daily_prices.csv, T years of OHLCV data for N tickers. (final file name may differ)
https://www.kaggle.com/datasets/iamspace/precog-quant-task-2026 
Split: The training, testing, and validations splits are entirely up to your creative process. However, keep in mind that we want to see your model perform out of sample for as long of a time frame as possible.
Note: The years of data and number of stocks in the file that will be provided is open to change. This shouldn’t affect how you approach the problem.

Part 1: Feature Engineering & Data Cleaning
Raw market data is messy and noisy. Show us how you handle it. Good data = good features, and good features = good model. (this prior statement has no mathematical backing and is not subject to any guarantees)
Cleaning: Assess data quality. Handle missing values, outliers, or potential anomalies in the provided dataset.
Feature Extraction: Generate features that capture market dynamics (e.g., momentum, volatility, volume patterns, or statistical factors). Get fancy with it. Think. What would actually be useful for a model to know?
Note: The quality of your inputs defines the ceiling of your performance. Simple raw prices are rarely sufficient for high-quality predictions. Too many features isn’t a good thing either. Quant research is fast, so your models must be faster. And remember, you are dealing with a universe here; not a single asset.
Deliverable:
A Python notebook showing your cleaning logic and feature generation code.

Part 2: Model Training & Strategy Formulation
Develop a predictive engine that translates your features from Part 1 into actionable trading signals for the universe.
Prediction: Build a model (or models) to predict asset performance or rank attractiveness. 
You have two broad choices for how you choose to define your prediction target:
Classification (whether the stock goes up, down, or stays the same)
Regression (predict, say, 1 day forward return, or 5 day forward return)
Regardless of what you choose, keep in mind that you will eventually have to convert this prediction into a viable, tradeable signal that your backtester is able to interpret.
Strategy Logic: Define how these predictions translate into portfolio decisions.
Hint A: Financial data is incredibly noisy. Relying on a single signal source or a single naive model architecture can often lead to instability. 
Hint B: Markets evolve. A relationship that held true in Year 1 may not exist in Year 3. Consider how your methodology ensures relevance as market dynamics shift over time.
Deliverable(s):
A notebook detailing your modeling approach and the logic used to generate predictions.

Part 3: Backtesting & Performance Analysis
Simulate your strategy over the 2-Year Test Set. Your backtester must realistically model the execution of your strategy.
Simulation Constraints:
Initial Capital: $1,000,000. (this isn’t necessary, you can use whatever)
Transaction Costs: Deduct 0.10% (10 bps) per trade. (play around with it, we also want to see performance in absence of transaction costs)
Universe: You may trade any subset of the N stocks.
Metrics: Report the following metrics (Note that all these metrics have multiple interpretations. Your choice of interpretation is also open to you, but you will be expected to justify your choice in the context of your pipeline):
Sharpe Ratio (annualized)
Maximum Drawdown (and average)
Portfolio Turnover
Return (how much money we made 💸)
Deliverable(s):
The backtesting code. We want to see out-of-sample performance on at least two years worth of data.
A cumulative PnL plot (Strategy vs. Equal-Weight Benchmark). How much did your strategy make over a benchmark? (either the market or an equal-weight buy-and-hold portfolio of the same assets)
A brief analysis: Did the strategy survive transaction costs? When and why did it fail?
Notes:
We will be judging you based on the above metrics, but we will also be judging the period over which those metrics were delivered. A lower Sharpe that remains stable over a long time frame is much more valuable than a ridiculously high Sharpe over one year. Always remember, you cannot see into the future.



Part 4: Statistical Arbitrage Overlay
While the main pipeline focuses on broad alpha, specific opportunities often exist in relative value.
Oftentimes, there exist assets that exhibit correlated or cointegrated movement: they mode together. This movement may be instantaneous, or it may occur after a certain time lag.
Your goal will be to find examples of such co-moving assets and explain the rationale behind their discovery.
Do not stick to pure correlation. Use it as a baseline, but get creative with it. I won’t provide any guidance or hints here as it would only serve to bias your research.
Which assets seem correlated? Do they lie in the same sector? Is their customer base in the same country? Does sentiment play a role? Which timeframes do they appear to move together in? Which assets leads the relationship, and which asset lags in it?
This is a very open ended question, and you may not even see statistically significant results. Do not be discouraged by this. It’s all part of the quant research process. Make visuals, present ideas. We want it all.
Deliverable(s):
Analysis: Present visuals and analysis of identified pairs or asset groups whose movements appear to be tethered together. Mathematically and empirically justify your approach and results.
Implementation Idea: Demonstrate how you would incorporate this relative-value signal into your main portfolio structure. You don’t need to code this up explicitly, but we would love to see mathematically backed ideas.

Something to consider before you dive in (feel free to skip)
Quant research is messy. From start to finish, many things are abstracted away, and at times, the mathematical formulations used by researchers to try and make sense of things seem to only drive you further away from the truth. This is completely natural! Take it in stride as you proceed with this project.
Don’t be discouraged by the lack of results, and always be wary of results that seem too good.
Remember that we cannot see into the future, and however elite your model is, it can’t either.
AI is your best friend that sometimes can’t be trusted. Use it for your research, use it to learn, but always, always, validate what it’s telling you.
Have fun with it! Explore the data, make your visualizations, see how things move. An intuitive understanding only serves to supplement a purely mathematical one.
The data is very naively anonymized. It’s actually really easy to reverse engineer and figure out which asset’s data it actually is. Feel free to try and come up with a mapping! It won’t help you with this task regardless.

Doubts Document Link: 
https://docs.google.com/document/d/1ybowfIuIkde2ggIqVEkyZ84t4IpuDXBiX0Wk5ZOZs3E/edit?usp=sharing

This is the initial context prompt, take this in very carefully this is the bible. We are gonna go through this very slowly comprehensively and detail our work and approach and methodlogy at every stage of the project. I want the entire workflow to be extremely organized and neat. We'll go through the stages step by step and try to emphasize our creativity and implementing radically different ideas and  approaches maybe well known ones or even cutting edge new papers, we'll learn the math if needed anywhere and also focus on Part 4 being extremely creative and unexpected. Remember this prompt. You are a quant researcher at Jane Street, follow the rigor. 


