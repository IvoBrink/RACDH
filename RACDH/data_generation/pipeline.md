# Pipeline for generating contextual vs. parametric
## 1. Detecting the proper entities
From a Wikipedia article we detect all entities. We remove the ones too similar to each other or the title of the passage (using cosine similarity of embeddings). The intuition behind this is that the entity is not too detrimental to the text, making it hard to remove any mention of it for the parametric task, but also preventing too much repitition making contextual retrieval trivial.

## 2. Knowing
First, for a *target model*, we determine if it knows the relation *article-entity* without any further context. We save this.

We want to test this *thoroughly*. So we design 3 completion-settings for every article-entity pair. 

### 2.1 Design completions
We prompt an instruct model to generate the completions using (i) the passage and (ii) the entity. We define 3 types.
* Bob-Alice.
* Truncate passage.
* In conversation question.

### 2.2 Generate completions
The *target model* then uses the completions

## 3. Removing (if necessary)
If the model knows the relationship we wish to test its parametric knowledge. This requires removing the entity from the article. If it doesn't, we will experiment with contextual capabilities by keeping the entity within the article

## 4. Completions
We now have articles paired with entities in which either its reference is removed (parametric) or where it is not (contextual). We prompt a model to reintroduce the entity, we do string matching to find it and truncate the text right up until the mention of the entity. We formulated a completion task. 
