---
created: 10 Jul 2025
---
Moderators should review the notebook before hand.
# Part I
Discussion on AI lifecycle, the process of creating an AI (in context of medical usage)

# Part II
Discussing *what fairness means*, i.e. not using unrelated features as assumption. 
## Discussion 1
What biases are there in this dataset?
- It only captures from a few countries.
- Might not be suitable for Thailand / Asian usage
- Only capture for one year, might contain seasonal effects (some diseases or complications that are unusually common in the year)

After showing the plots, briefly discuss that the mean age is high, younger people are under-represented. White people flooded the dataset. And men are more susceptible to most diseases.

We then begin introduction to the main problem, making an ML model on cirrhosis. 

## Discussion 2
Think of a problem and what data do we need. How to collect those with no biases. For example, if we choose *detecting lung cancer from CXR*, then we collect CXR images from people. We might also want to know the age, gender, or if they smoke.
> Even when training to detect using only from CXR, knowing if the person smokes can help reducing the bias during the training.

# Part III
We discuss confusion matrix and the three common metrics (accuracy, precision, recall). You should be able (if question arises) to explain that high accuracy is not always good ([see this](https://www.facebook.com/share/p/1Ak6oXcB4n/)) In cases like in medical field, we *usually aim* for recall.

Running the cells will produce plots that give insights on model performance. I think giving a short discussion after the first cell, asking if this result is good enough might be useful. We then answer that it is not using the second plot, treating all as male.

Bias mitigation is based on the *medical fact* that gender doesn’t **biologically** affect cirrhosis, but **culturally**, men drink more. We resampled the data to equally represent the two genders in order to remove the bias. Now, having another column, *drink* that captures how much do people drink will help improve the model, but thats out of scope.

## Discussion 3
IMO, this is the most important discussion.
- Can biases help the model?
	- Yes! Gender bias did make the model more accurate in the previously discussed problem.
- How do you know if bias is fair or unfair, good or bad?
	- To me, fair bias is a feature.
	- In the case of cirrhosis, we understand that there is no biological different, so this is an *unfair bias*. If we change cirrhosis to a different disease that we don’t understand, then gender bias might be fair or unfair (i don’t know). 
	- The key is that you need to understand the problem and how the feature should affect the decision. (like you understand that gender *should not* be a feature, so it is a (bad) bias)
- How can we prevent (bad) biases?
	- Having a good uniformly distributed data.
	- Represent all groups equally in training data.
- Is chatGPT safe from biases? What questions should we asked?
	- GPT is similarly trained, one shouldn’t expect it to be fair
	- “Draw a picture of a nurse” should already give a picture of a female nurse.
	- I believe workshop 2 also tackle this (?)
- chatGPT has bias, then can we still use it? How?
	- with cautious, of course :)

## Potential Question
- Why can't we train the model without using gender. 
	- Training the model without using gender is still not fair in the same way. Surely, the model cannot *explicitly* discriminate against gender, but it can in a more subtle ways. 
	- One way the model do that be inferring the gender based on height, as statistically men are taller, we then get an *implicitly biased* model. 
	- It doesn't fix the problem, just make it invisible. 