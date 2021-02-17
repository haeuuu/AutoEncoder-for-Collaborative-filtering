# AutoEncoder for Collaborative filtering

AutoEncoder 구조를 기반으로, 다음 두가지 모델을 실험해보았습니다.

1. AutoEncoder로 Content based, Collaborative filtering 모델을 각각 학습하여 추천합니다.

   NOTE : Collaborative Filtering에 Content based Recommendation 결과를 "결합"하여 sparsity 해결을 시도합니다.

   ```python
   Music nDCG: 0.103753
   Tag nDCG: 0.302064
   Score: 0.133499
   ```

2. song을 user로, user를 특성으로 보고 song embedding을 학습합니다.

   ```python
   Music nDCG: 0.187719
   Tag nDCG: 0.387189
   Score: 0.217639
   ```

<br></br>

## :a: AutoEncoder 기반의 Content Based & Collaborative Filtering



#### Introduction

일반적으로 CB와 CF를 함께 쓴 추천 시스템이라고 한다면 CB로 일부를, CF로 일부를 추천한 후 합쳐서 제시한다.

그러나 CF 모델에 넣기에는 **playlist가 가진 정보가 부족해서 노래를 제대로 추천해주지 못하는 경우**가 생겼다.

CB만으로 추천한 모델보다는 CF를 이용한 모델이 점수가 더 높은데, CF는 tag 점수가 너무 약했다. 모델은 바꾸지 않고 << 태그 점수도 올리면서 + music nDCG까지 올리고 + sparse함을 해결할 방법 >> 을 생각해보다가  **CB를 통해 노래를 추천한 후, 이를 query에 결합하여 CF의 input으로 사용해보자** ! 하는 아이디어가 떠올랐다.

단순히 CB와 CF를 합친 것 보다 좋은 성능을 발휘했다.

<br></br>

### 1 ) Content-based Recommendation

AutoEncoder를 이용하여 1024차원의 song embedding을 학습하고 cosine similarity를 이용하여 유사한 노래를 찾자.

![image-20201108011245060](fig/image-20201108011245060.png)

1. 발매 년도, 발매 월, 장르, 고빈도 태그를 이용하여 457차원의 one-hot vector를 생성한다.

2. `457 > 512 > 512 > 1024 > 512 > 512 > 457` 으로 **차원을 확장하였다가 줄이는 구조**를 가진 AutoEncoder를 이용하여 학습한다.

   * decoder의 가중치는 encoder 가중치를 전치하여 그대로 이용한다.

   ```python
   non_linearity_type = "relu"
   drop_prob = 0.8
   weight_decay = 0.0001
   lr = 0.0001
   num_epochs = 50
   batch_size = 128
   
   optimizer = optim.Adam()  
   loss = loss1 + loss2*3  where loss1 = customedMSEloss(inputs, outputs)
   							  loss2 = nn.MSEloss(inputs, outputs, reduction = 'mean')
   ```

3. encoder를 통해 song embedding을 추출한 후 cosine similarity를 계산하여 가장 유사한 노래부터 내림차순으로 정렬하여 저장한다.

<br></br>

## 2 ) Collaborative Filtering

![image-20201108012520829](fig/image-20201108012520829.png)

1. 약 27만개의 곡을 이용하여 one hot vector를 생성한다.

2. `277,218 > 1024 > 277,218` 구조의 AutoEncoder를 이용하여 학습시킨다.

   * encoder와 decoder 가중치는 독립적으로 학습한다.

   ```python
   non_linearity_type = "selu"
   drop_prob = 0.8
   weight_decay = 0
   lr = 0.2
   num_epochs = 45
   batch_size = 1024
   
   optimizer = optim.Adam()  
   loss = loss1 + loss2*50  where loss1 = customedMSEloss(inputs, outputs)
   						 	   loss2 = nn.MSEloss(inputs, outputs, reduction = 'mean')
   ```

<br></br>

### NOTE

* #### Loss design하기

  논문에서 제시한대로 `Masked MSE`만을 이용하여 모델을 학습시켜보았으나 아무리 epoch을 늘려도 학습이 되고 있지 않았다.

  이유를 파악하기 위해 model의 output을 보니, **모든 값을 1로 예측**하고 있었다.

  `Masked MSE`만을 loss로 사용하게 되면 1을 1로 잘 맞추는 것이 중요해지기 때문에 **"그냥 모든 값을 1로 예측하자!"가 되어버린다**!!

  그러므로 **0을 모두 1로 예측하지 않도록 만들기 위해 일반적인 MSE**도 더하여 최종 loss로 사용한다.

  모델에 따라 weight은 달라진다.

  ```python
  def customedMSEloss(inputs, targets, size_average=False):
      # Masked MSE
  	mask = targets != 0
      num_ratings = torch.sum(mask.float())
      criterion = nn.MSELoss(reduction='sum' if not size_average else 'mean')
      return criterion(inputs * mask.float(), targets), Variable(torch.Tensor([1.0])) if size_average
  			else num_ratings
      
  loss1 = customedMSEloss(inputs, targets)
  loss2 = 일반 MSE
  total_loss = w1*loss1 + w2*loss2
  ```

  <br></br>

* #### Relu v.s. Selu

  activation function이 어떤 영향을 끼칠까?

  똑같은 조건에서 activation function만 다르게 하여 30epoch을 돌린 후 비교해보자.

  * ##### relu를 사용한 경우

    * 원래 1인 item을 1로 잘 예측하지 못한다. (원래 1인 item의 절반 이하만 맞춘다.)
    * 원래 0인 item중 새롭게 추천할 item (0.5 이상으로 예측한 item)의 수가 selu에 비해 적다.

    ```python
    # example
    
    > i-th song을 포함한 playlist 수 					 : 483개 (즉 i-th song에 대한 input에 1이 483개)
    > i-th song을 포함한 playlist 중 제대로 예측한 것의 갯수  : 213개
    > i-th song이 어울릴 것이라고 새롭게 예측한 playlist수    : 256개
    ```

    정답도 제대로 못맞추는데, 새로 추천한 playlist를 믿을 수 있을까? NO!
    <br></br>

  * ##### selu를 사용한 경우

    * 원래 1인 item을 1로 잘 예측한다. (원래 1인 item의 2/3 정도를 맞춘다.)
    * 원래 0인 item중 새롭게 추천할 item의 수가 

    ```python
    # example
    
    > i-th song을 포함한 playlist 수 					 : 483개 (즉 i-th song에 대한 input에 1이 483개)
    > i-th song을 포함한 playlist 중 제대로 예측한 것의 갯수  : 353개
    > i-th song이 어울릴 것이라고 새롭게 예측한 playlist수    : 246개
    ```

    원래 input을 더 잘 복원하고 있음 ! 새롭게 추천한 playlist도 더 믿을만하지 않을까? 선택하자!

  * 근데, 왜 이런 차이가 생겼을까?

    * 논문에서도 그저 'negative part를 죽이지 않는 것이 중요하다.' 정도로만 언급했다.

    * 내가 생각한 이유는!

      relu처럼 음수 부분을 모두 0으로 만들어버리면 **다음 두 정보가 모두 0으로 통일**된다.

      * `score < 0` : "이 playlist 와는 **정말 어울리지 않아!**"
      * `score = 0.001` : "이 playlist에는 .. **들어갈 수도 있고 아닐 수도 있겠네**"

      이렇게 되면, <호>에 대한 정보는 학습할 수 있지만 **<진짜 진짜 불호>에 대한 정보는 학습할 수 없게 된다**!
      어떤 playlist들이 서로 상극인지를 안다면 특성을 더 잘 파악할 수 있을 것이다.

<br></br>

### Recommendation

> sparse한 validation을 **Content-based recommendation을 통해 약간 더 dense하게 만들어 준 뒤 CF를 진행**합니다.

![image-20201108004704580](fig/image-20201108004704580.png)

1. Query를 one hot vector로 만든다.

2. Query에 포함된 모든 곡에 대하여 각 1개 ~ 3개의 유사곡을 찾아 추천한다. `Content_based_recommendation`

3. `Content_based_recommendation`에 속한 곡을 이용하여 **Query vector를 채운다**. `Query*`

   > query에는 정보가 부족한 경우가 많다.
   >
   > **Content based로 추천한 결과를 Query에 더해줌으로써 CF 추천 결과의 성능을 높이고자 하였다.** 
   >
   > 3번 과정을 통해 Query 내에 약 100개의 노래가 포함되도록 한다.

4. `Query*`를 AutoEncoder에 넣어 점수를 내림차순 한 후 노래를 추천한다.

5. 추천한 곡들과 가장 많이 매칭된 tag를 찾아 10개를 추천한다.



* 만약 Query에 노래가 없을 경우
  1. tag가 있다면 해당 tag와 많이 매칭된 곡을 찾아 100곡을 추천한다.
  2. tag가 없다면 train set에서 가장 많이 등장한 100곡을 찾아 추천한다.

<br></br>

### Score

```python
Music nDCG: 0.103753
Tag nDCG: 0.302064
Score: 0.133499
```

<br></br>

:raising_hand_woman: **둘 중 하나만을 이용하거나, 단순히 섞기만 해서는 위와 같은 점수를 낼 수 없다.**

Content-based의 결과를 Query vector에 포함해서 CF model에 넣는 방식이 미미하지만 성능 향상에 영향을 끼쳤다고 볼 수 있다.

1. **CF만을 이용해서 추천했을 경우** :  노래는 잘 맞추지만 태그 성능이 좋지 않음.

   ```python
   Music nDCG: 0.0984524
   Tag nDCG: 0.17614
   Score: 0.110106
   ```

2. **CB만을 이용해서 추천했을 경우** :  tag는 잘 맞추지만 노래 점수는 크게 하락함.

   ```python
   Music nDCG: 0.0675142
   Tag nDCG: 0.338713
   Score: 0.108194
   ```

3. **두 방법을 단순히 함께 이용한 경우** : 30곡 이하이면 2곡씩, 30곡 이상이면 1곡씩 CB로 추천, 나머지는 CF로 추천

   * CB로 많이 추천해줄 수록 tag 점수는 올라가지만 노래 점수는 하락함.
   * 노래 점수는 CF만을 쓴 것 이상으로 올라갈 수 없었음.

   ```python
   Music nDCG: 0.0890984
   Tag nDCG: 0.282032
   Score: 0.118038
   ```

  <br></br>

## :b: meta 정보와 user에 의해 생성된 정보를 함께 넣어 적합시킨 AutoEncoder



#### Introduction

* 노래 embedding 자체에 user들이 제공한 정보를 포함할 수는 없을까?

* 장르가, 태그가 비슷한 노래 뿐만 아니라 meta 정보는 다를지라도 사용자들이 함께 듣는 노래들을 묶어주면 어떨까?

하는 아이디어에서 출발하였습니다.



논문의 conclusion에서는 이 모델을 item에 쓰는 것 보다 user에 쓰는 것을 추천하고 있습니다.

생각을 조금 전환하여 **song을 user로, meta 정보와 playlist에 대한 정보를 item으로** 생각할 수 있겠다고 판단하였습니다.

만약 song1, song3, song10이 하나의 play list에 있다면, **함께 있는 song들을 "서로 취향이 비슷한 user"**라고 판단합니다.

**song1과 유사한 노래는, 비슷한 취향을 가진 song3에게 추천해주어도 좋아할것이다! 라는 가정**을 세웠습니다.

song embedding을 통해 유사한 노래를 찾고, playlist에 포함된 노래에서 유사곡을 k곡씩 골라 100곡을 채워 추천합니다.

<br></br>

### Model

```python
layer_size = one_hot_encoding.shape[1]
hidden_layers = 24,684 > 2,048 > 1,024 > 2,048 > 24,684
non_linearity_type = "selu"
drop_prob = 0.8
weight_decay = 0.0001
lr = 0.0005
num_epochs = 50
batch_size = 2048
```

1. 논문에서 제시한대로 **negative part를 없애지 않는 selu**를 activation fuction으로 사용

   * epoch 30에서 relu를 사용한 모델과 비교한 결과 selu를 사용한 모델이 `True => True` 로 더 잘 예측하였음.

   * 예측 값이 0.5 이상일 경우 고득점으로 생각하고 어느정도 정답을 맞추었다고 가정한 후 평가.

     Relu를 사용한 경우에 1을 0 혹은 0.00x로 예측한 경우가 많았음. 1을 0.5 이상으로 맞춘 경우 역시 대부분 sample에서 50%를 넘지 못함.

     그러나 Selu를 사용한 경우에는 1을 0.5 이상으로 예측한 경우가 약 60%정도로 보임. 더 잘 학습되었다고 판단함.

2. Loss function을 수정

   * loss는 **Masked MSE** 이외에 일반적인 MSE를 wieghted sum해서 사용하였음 **(w1 = 15, w2 = 200)**
   * 학습 초기에는 MMSE가 더 높으나(즉 학습 초반에는 1을 1로 잘 예측하지 못하고 있음.) 두 loss가 동시에 낮아지면서 차이가 미미해짐. 1을 맞추면서도 0을 모두 1로 예측하지 않도록 균형을 잡으며 학습할 것으로 기대하였음.
   
3. Re feeding을 하지 않았음.

   * Re feeding을 하는 가정은 <u>잘 학습된 AE의 경우 `AE(x*) = AE(x)`를 만족해야 한다</u>임.

     > 자세한 내용은 논문, 혹은 [논문 리뷰](https://github.com/haeuuu/RecSys-PaperReview/blob/master/Training_DeepAE_for_CF.md) 참고

   * 그러나 **song을 user로 보게 된다면 이를 만족하지 않아야 한다**고 생각하였음. 

   * 같은 장르, 같은 태그를 가졌다고 할지라도 어떤 play list에 담겨있는가를 통해 다른 embedding을 가져야 하는 것이 더 좋은 모델이라고 판단
   
     ```python
     # tag는 같아도 embedding은 달라지면 좋을 예시
     
     > song1에 달린 tag : [지친하루, 피로]
         	 포함된 playlist 제목 : 위로가 되는 잔잔한 힐링송
     
     > song2에 달린 tag : [지친하루, 피로]
              포함된 playlist 제목 : 신나는 락으로 피로 날리자
     ```

<br></br>

### 유사곡 추출

cosine similarity를 이용하여 자기 자신을 제외한 유사곡을 100곡씩 추출하여 저장



**아이유의 봄 사랑 벚꽃 말고와 유사한 곡 top5**

![image-20201110185916959](fig/image-20201110185916959.png)



**오마이걸의 비밀정원과 유사한 곡 top5**

![image-20201110185946225](fig/image-20201110185946225.png)



**장범준의 노래방에서와 유사한 곡 top5**

![image-20201110190003554](fig/image-20201110190003554.png)

<br></br>

## Recommendation

1. query에 담긴 song embedding을 모두 더한다. `query_ply_embedding`

2. `query_ply_embedding`과 유사한 playlist train data에서 찾는다. (cosine simliarity)

3. 상위 50개 playlist에 달린 태그를 빈도 내림차순으로 정렬하여 10개를 뽑는다.

   상위 10개 playlist에 달린 song을 빈도 내림차순으로 정렬하여 100개를 뽑는다.

<br></br>

## Score

```python
Music nDCG: 0.187719
Tag nDCG: 0.387189
Score: 0.217639
```



#### Reference

* [Training Deep AutoEncoders for Collaborative Filtering](https://arxiv.org/pdf/1708.01715.pdf)