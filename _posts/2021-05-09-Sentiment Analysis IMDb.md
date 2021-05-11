```python
import pandas as pd
import numpy as np
import nltk
import timeit
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
```


```python
df = pd.read_csv("imdb.csv")
```


```python
# positive sentiment = 0, negative sentiment = 1
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>China White (1989) was Ronny Yu's first intern...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alejandro Amenabar, the young and talented Spa...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Yes, I watch this show. Because my girlfriend ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I've loved all of Cream's work, even as there ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The remake of H.B. Halicki's classic seventies...</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['review'].dropna(inplace=True)
```


```python
df['review'] = [entry.lower() for entry in df['review']]
```


```python
df['review'] = [word_tokenize(entry) for entry in df['review']]
```


```python
print(df['review'])
```

    0       [china, white, (, 1989, ), was, ronny, yu, 's,...
    1       [alejandro, amenabar, ,, the, young, and, tale...
    2       [yes, ,, i, watch, this, show, ., because, my,...
    3       [i, 've, loved, all, of, cream, 's, work, ,, e...
    4       [the, remake, of, h.b, ., halicki, 's, classic...
                                  ...                        
    9995    [i, enjoyed, carax, 's, ``, les, amants, du, p...
    9996    [as, a, helpful, warning, for, others, ,, i, b...
    9997    [i, am, a, fan, of, his, ..., this, movie, suc...
    9998    [loved, today, 's, show, !, !, !, it, was, a, ...
    9999    [this, documentary, is, the, most, hypnotizing...
    Name: review, Length: 10000, dtype: object
    


```python
print("- Splitting dataset (80% train, 20% test)...")
train_x, test_x, train_y, test_y = model_selection.train_test_split(df['review'], df['sentiment'], test_size=0.2)
print("- Data splitting is complete.")
print(train_x)
print(train_y)
```

    - Splitting dataset (80% train, 20% test)...
    - Data splitting is complete.
    3331    [bad, ,, bad, ,, bad, ., how, do, movies, like...
    5006    [this, is, so, poor, it, 's, watchable., <, br...
    7117    [ok., first, said, ,, i, just, wanted, to, che...
    3048    [it, was, a, serious, attempt, to, show, the, ...
    8887    [i, tried, to, watch, this, movie, in, a, mili...
                                  ...                        
    118     [marked, for, death, (, 1990, ), spends, more,...
    5579    [a, stupid, young, man, becomes, obsessed, wit...
    4754    [night, of, the, living, homeless, is, a, funn...
    7760    [i, would, of, given, this, film, a, zero, out...
    698     [the, cast, is, excellent, ,, the, acting, goo...
    Name: review, Length: 8000, dtype: object
    3331    negative
    5006    negative
    7117    negative
    3048    positive
    8887    negative
              ...   
    118     negative
    5579    positive
    4754    positive
    7760    negative
    698     positive
    Name: sentiment, Length: 8000, dtype: object
    


```python
Encoder = LabelEncoder()
print("Original Train_Y: ", train_y)
train_y = Encoder.fit_transform(train_y)
print("New Train_Y:      ", train_y)
print("Original Test_Y:  ", test_y)
test_y = Encoder.fit_transform(test_y)
print("New Test_Y:       ", test_y)
print("\n- Data encoding is complete.")

```

    Original Train_Y:  3331    negative
    5006    negative
    7117    negative
    3048    positive
    8887    negative
              ...   
    118     negative
    5579    positive
    4754    positive
    7760    negative
    698     positive
    Name: sentiment, Length: 8000, dtype: object
    New Train_Y:       [0 0 0 ... 1 0 1]
    Original Test_Y:   3885    negative
    6396    positive
    497     positive
    3825    negative
    6058    negative
              ...   
    7633    negative
    5234    negative
    6319    positive
    726     positive
    6865    positive
    Name: sentiment, Length: 2000, dtype: object
    New Test_Y:        [0 1 1 ... 1 1 1]
    
    - Data encoding is complete.
    


```python

from sklearn.feature_extraction.text import TfidfVectorizer

def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)

tfidf.fit(df['review'])
tfidf.vocabulary_
```




    {'china': 13342,
     'white': 68351,
     '(': 1671,
     '1989': 2660,
     ')': 1672,
     'was': 67611,
     'ronny': 53229,
     'yu': 69803,
     "'s": 1282,
     'first': 24712,
     'international': 33088,
     'film': 24454,
     '.': 1924,
     'this': 62619,
     'u.k': 64781,
     ',': 1679,
     '/holland/hong': 2054,
     'kong': 35468,
     'production': 49187,
     'shot': 56393,
     'in': 32106,
     'english': 21990,
     'and': 5568,
     'slightly': 57395,
     'edited': 21173,
     'for': 25334,
     'the': 62372,
     'western': 68153,
     'audience': 7090,
     'american': 5388,
     'wong': 68988,
     'brothers': 10889,
     'michael': 40115,
     'russell': 53628,
     'were': 68122,
     'supposed': 60854,
     'to': 63168,
     'star': 59221,
     'together': 63215,
     'but': 11365,
     'due': 20635,
     'prior': 49046,
     'commitments': 14751,
     'unavailable': 65042,
     'so': 57853,
     'another': 5814,
     'actor': 4164,
     'steven': 59514,
     'leigh': 36520,
     'took': 63378,
     'his': 30355,
     'spot': 58871,
     'several': 55724,
     'hong': 30728,
     'stars': 59280,
     'such': 60443,
     'as': 6659,
     'tommy': 63301,
     'playing': 47659,
     'a': 3612,
     'mute': 42104,
     'always': 5282,
     'creepy': 16427,
     'william': 68597,
     'ho': 30456,
     'appear': 6166,
     'well': 67966,
     'director': 19090,
     'an': 5502,
     'interesting': 33036,
     'cameo': 11710,
     'spot.': 58874,
     '<': 3564,
     'br': 10408,
     '/': 2024,
     '>': 3573,
     'chow': 13475,
     'are': 6391,
     'holland': 30561,
     'run': 53574,
     'family': 23687,
     'business': 11331,
     'they': 62519,
     'want': 67489,
     'go': 27402,
     'legit': 36500,
     'other': 45013,
     'asian': 6713,
     'gangs': 26508,
     'do': 19722,
     "n't": 42230,
     'see': 55104,
     'there': 62476,
     'move': 41587,
     'face': 23482,
     'saving': 54262,
     'please': 47694,
     '``': 3611,
     'foreigners': 25395,
     "''": 6,
     'keep': 34865,
     'on': 44495,
     'making': 38429,
     'money': 41088,
     'old': 44412,
     'fashion': 23880,
     'way': 67760,
     'others': 45018,
     'take': 61514,
     'their': 62401,
     'even': 22651,
     'new': 42868,
     'low': 37721,
     'by': 11421,
     'smuggling': 57670,
     'drugs': 20538,
     'what': 68211,
     'not': 43627,
     'after': 4568,
     'elder': 21418,
     'is': 33486,
     'gunned': 28527,
     'down': 20152,
     'cold': 14412,
     'blood': 9622,
     'make': 38399,
     'against': 4614,
     'any': 6043,
     'who': 68397,
     'with': 68823,
     'them': 62412,
     'can': 11785,
     'families': 23684,
     'from': 25985,
     'killing': 35123,
     'one': 44548,
     'or': 44798,
     'while': 68288,
     'streets': 59946,
     'of': 44213,
     'flow': 25111,
     'innocence': 32717,
     'gangsters': 26515,
     '?': 3574,
     'highly': 30192,
     'underrated': 65284,
     'movie': 41599,
     'i': 31510,
     'surprised': 60934,
     'how': 31087,
     'good': 27592,
     'it': 33570,
     'have': 29419,
     'seen': 55138,
     'version': 66650,
     'that': 62349,
     'would': 69187,
     'be': 8311,
     'huge': 31176,
     'treat': 63997,
     'me': 39509,
     'longer': 37460,
     'has': 29320,
     'lot': 37607,
     'more': 41331,
     'cinema': 13634,
     'involved': 33339,
     'too': 63358,
     'bad': 7646,
     'available': 7247,
     'd.v.d': 17121,
     'video': 66779,
     'print': 49040,
     'saw': 54276,
     'washed': 67622,
     'out': 45058,
     'sound': 58318,
     'reproduction': 52059,
     'great.': 28034,
     'recommended': 51239,
     'action': 4110,
     'fans.': 23748,
     'factoid': 23519,
     ':': 3562,
     'takes': 61532,
     'place': 47525,
     'before': 8543,
     'decriminalization': 17813,
     'alejandro': 4943,
     'amenabar': 5375,
     'young': 69755,
     'talented': 61555,
     'spanish': 58441,
     'clearly': 13902,
     'shows': 56485,
     'us': 66141,
     'he': 29500,
     'serious': 55620,
     'maker': 38416,
     'anyone': 6057,
     'doubting': 20124,
     'should': 56406,
     'look': 37486,
     'at': 6913,
     'latest': 36135,
     'sea': 54920,
     'inside': 32785,
     'been': 8507,
     'rewarded': 52535,
     'numerous': 43851,
     'accolades': 3949,
     'only': 44630,
     'spain': 58427,
     'throughout': 62799,
     'world': 69108,
     'wherever': 68277,
     'wonderful': 68972,
     'shown.': 56483,
     'if': 31666,
     'you': 69741,
     'perhaps': 46726,
     'like': 36950,
     'stop': 59714,
     'here.': 29940,
     'ramon': 50522,
     'sampedro': 53997,
     'man': 38548,
     'confined': 15182,
     'bed': 8468,
     'being': 8625,
     'quadriplegic': 50049,
     'depends': 18274,
     'kindness': 35172,
     'strangers': 59869,
     'everything': 22741,
     'since': 56892,
     'accident': 3937,
     'thinks': 62582,
     'thing': 62556,
     'alone': 5182,
     'end': 21881,
     'life': 36844,
     '!': 0,
     'moral': 41301,
     'issue': 33559,
     'center': 12634,
     'story': 59757,
     'based': 8130,
     'real': 50993,
     'life.': 36868,
     'mr': 41734,
     'tells': 62045,
     'point': 47886,
     'view': 66825,
     'nothing': 43685,
     'here': 29937,
     'false': 23655,
     'manipulative': 38677,
     'part': 46054,
     'all': 5018,
     'relies': 51725,
     'facts': 23529,
     'known': 35405,
     'country': 16064,
     'case': 12268,
     'became': 8444,
     'cause': 12479,
     'celebre': 12572,
     'favor': 23995,
     'euthanasia': 22618,
     'theme': 62426,
     'no': 43208,
     'wanted': 67495,
     'deal': 17601,
     'spain.': 58428,
     'its': 33629,
     'background': 7599,
     'predominantly': 48728,
     'roman': 53172,
     'catholic': 12441,
     'evolved': 22798,
     'into': 33186,
     'most': 41445,
     'democratic': 18139,
     'societies': 57925,
     'europe': 22605,
     'distinction': 19579,
     'notable': 43663,
     'because': 8445,
     'long': 37437,
     'years': 69600,
     'dominated': 19904,
     'dictator': 18788,
     'yet': 69658,
     'spite': 58745,
     'advances': 4419,
     'society': 57927,
     'idea': 31599,
     'taking': 61536,
     'own': 45529,
     'something': 58125,
     'understood': 65311,
     'majority': 38392,
     'citizens': 13723,
     'still': 59569,
     'considered': 15359,
     'subject': 60292,
     'could': 16000,
     'done': 19943,
     'country.': 16069,
     'loved': 37692,
     'lived': 37220,
     'intense': 32968,
     'when': 68263,
     'enlisted': 22051,
     'sailor': 53876,
     'discover': 19278,
     'having': 29435,
     'him': 30261,
     'lands': 35954,
     'experience': 23146,
     'cultures': 16865,
     'love': 37661,
     'affair': 4486,
     'people': 46611,
     'galicia': 26410,
     'learn': 36361,
     'childhood': 13301,
     'imagine': 31817,
     'same': 53974,
     'friendly': 25912,
     'away': 7352,
     'knew': 35336,
     'second': 55015,
     'goes': 27498,
     'vibrant': 66716,
     'vegetable': 66504,
     'shattered': 56024,
     'suddenly': 60472,
     'must': 42070,
     'leave': 36388,
     'aside': 6720,
     'care': 12054,
     'home': 30624,
     'brother': 10875,
     'sister-in-law': 57011,
     'stoic': 59651,
     'situation': 57042,
     'matter': 39214,
     'fact': 23512,
     'lives': 37233,
     'become': 8461,
     'afterthought': 4596,
     'comes': 14640,
     'tend': 62110,
     'sick': 56618,
     'without': 68846,
     'protesting': 49481,
     'blaming': 9446,
     'sacrifices': 53770,
     'alive.': 5016,
     'why': 68453,
     'minds': 40484,
     'sampedros': 53998,
     'ca': 11499,
     'comprehend': 14956,
     'wishes': 68789,
     'given': 27233,
     'up': 65985,
     'normal': 43550,
     'weighs': 67918,
     'heavily': 29674,
     'these': 62503,
     'uncomplicated': 65127,
     'simple': 56846,
     'doing': 19856,
     'came': 11702,
     'naturally.': 42492,
     'legal': 36469,
     'meaning': 39548,
     'enter': 22126,
     'desire': 18425,
     'help': 29832,
     'put': 49967,
     'suffering': 60490,
     'julia': 34449,
     'lawyer': 36271,
     'herself': 30013,
     'handicapped': 29031,
     'suffers': 60493,
     'rare': 50659,
     'malady': 38443,
     'rosa': 53281,
     'fish': 24738,
     'cannery': 11856,
     'worker': 69086,
     'becomes': 8464,
     'infatuated': 32513,
     'javier': 33912,
     'bardem': 8011,
     'makes': 38419,
     'brilliant': 10720,
     'transformation': 63832,
     'total': 63538,
     'we': 67793,
     'doubt': 20118,
     'moment': 41043,
     'else': 21588,
     'paralyzed': 45929,
     'mr.': 41736,
     'use': 66159,
     'order': 44837,
     'convey': 15646,
     'emotions': 21758,
     'trapped': 63915,
     'best': 8916,
     'role': 53123,
     'career': 12060,
     'surpasses': 60927,
     'award': 7341,
     'winning': 68723,
     'performance': 46699,
     'reynaldo': 52562,
     'arenas': 6401,
     'late': 36111,
     'cuban': 16786,
     'poet': 47868,
     'portrayed': 48248,
     'night': 43057,
     'falls': 23651,
     'supporting': 60848,
     'roles': 53129,
     'belen': 8651,
     'rueda': 53509,
     'impressive': 32058,
     'appearance': 6168,
     'woman': 68935,
     'fighting': 24410,
     'her': 29914,
     'physical': 47167,
     'problems': 49128,
     'lola': 37398,
     'duenas': 20641,
     'also': 5218,
     'effective': 21238,
     'kindred': 35174,
     'soul': 58304,
     'loves': 37709,
     'deeply': 17858,
     'celso': 12610,
     'bugallo': 11073,
     'crossroads': 16645,
     'mabel': 38075,
     'rivera': 52897,
     'compassionate': 14834,
     'manuela': 38743,
     'never': 42849,
     'asks': 6734,
     'anything': 6060,
     'tends': 62122,
     'questioning': 50155,
     'she': 56049,
     'all.': 5063,
     'composed': 14939,
     'haunting': 29407,
     'music': 42025,
     'score': 54679,
     'cease': 12542,
     'surprise': 60931,
     'wonders': 68984,
     'next': 42938,
     'project': 49286,
     'will': 68584,
     'success': 60419,
     'whatever': 68225,
     'might': 40269,
     'decide': 17744,
     'future': 26285,
     'yes': 69649,
     'watch': 67664,
     'show': 56433,
     'my': 42150,
     'girlfriend': 27190,
     'watches': 67683,
     'course': 16096,
     'least': 36374,
     'tell': 62035,
     'friends': 25915,
     'nobody': 43265,
     'ever': 22676,
     'say': 54292,
     ';': 3563,
     'excellent': 22915,
     'trash': 63921,
     'tv': 64540,
     'panel': 45809,
     'tyra': 64752,
     'does': 19811,
     'miss': 40764,
     'single': 56931,
     'opportunity': 44754,
     'talk': 61568,
     'about': 3800,
     'taken': 61524,
     'seriously': 55622,
     'quite': 50226,
     'businesswoman': 11345,
     'then': 62436,
     'jay': 33923,
     'manuel': 38742,
     'sober': 57896,
     'gay': 26688,
     'guy': 28599,
     'very': 66667,
     'amazingly': 5327,
     'professional': 49215,
     'extroverted': 23371,
     'queen': 50128,
     'catwalk': 12468,
     'damn': 17228,
     'funny': 26214,
     'nigel': 43046,
     'self-styled': 55326,
     'panel.': 45810,
     'in-depth': 32114,
     'shallow': 55915,
     'industry': 32453,
     "'ve": 1582,
     'got': 27762,
     'give': 27226,
     'producers': 49178,
     'showing': 56479,
     'which': 68283,
     'informative': 32587,
     'entertaining.': 22147,
     'finally': 24591,
     'edition': 21179,
     'great': 28028,
     'just': 34528,
     'right': 52744,
     'mix': 40852,
     'girls': 27199,
     'living': 37242,
     'ego-house': 21304,
     'impressions': 32057,
     'sets': 55683,
     'competitions': 14868,
     'themselves': 62433,
     'judging': 34413,
     'slides': 57389,
     'smoothly': 57654,
     'beat': 8386,
     'lenghts.': 36563,
     'factors': 23524,
     'led': 36425,
     'eliminations': 21517,
     'made': 38198,
     'capricious': 11960,
     'divas': 19650,
     'quick': 50170,
     'bitching': 9309,
     'woe': 68900,
     'those': 62666,
     'offends': 44268,
     'said': 53863,
     'contestant': 15502,
     'step': 59447,
     'toes': 63208,
     'judgment': 34414,
     'usually': 66192,
     'fair': 23583,
     '-': 1695,
     'think.': 62572,
     'your': 69770,
     'brain': 10437,
     'break': 10551,
     'join': 34251,
     'time': 62968,
     'watching': 67686,
     'bit': 9303,
     'reality': 51019,
     'ones': 44616,
     'crave': 16308,
     'somehow': 58112,
     'cream': 16346,
     'work': 69073,
     'small': 57542,
     'precious': 48668,
     'catalog': 12372,
     'hold': 30534,
     'twenty': 64606,
     'minutes': 40596,
     'some': 58106,
     'songs': 58180,
     'spoonful': 58852,
     'toad': 63177,
     'off': 44219,
     'wheels': 68260,
     'fire': 24668,
     'prime': 49011,
     'examples': 22898,
     'rock': 53031,
     'socks': 57954,
     'than': 62329,
     'half': 28818,
     'act': 4086,
     'working': 69091,
     'today': 63191,
     'power': 48474,
     'gel': 26759,
     'stage': 59078,
     'anticipate': 5970,
     'band': 7891,
     'reunions': 52400,
     'royal': 53437,
     'albert': 4904,
     'hall': 28889,
     'last': 36093,
     'year': 69587,
     'may': 39290,
     'gotten': 27783,
     'older': 44434,
     'fans': 23746,
     'energy': 21950,
     'arrangements': 6542,
     'classic': 13823,
     'blues': 9726,
     'renditions': 51885,
     'room': 53241,
     'badge': 7684,
     'politician': 47991,
     'sunshine': 60662,
     'seems': 55136,
     'clapton': 13790,
     'solos': 58085,
     'formation': 25484,
     'sometimes': 58134,
     'solo': 58079,
     'ginger': 27148,
     'baker': 7760,
     'enough': 22070,
     'jack': 33720,
     'bruce': 10912,
     'sturdy': 60196,
     'vocals': 67109,
     'kind': 35157,
     'get': 26976,
     'bottom': 10265,
     'line': 37053,
     'wish': 68784,
     'had': 28715,
     'dvd': 20842,
     'cool': 15690,
     'special': 58525,
     'features': 24079,
     'remake': 51776,
     'h.b': 28664,
     'halicki': 28886,
     'seventies': 55716,
     'chase': 12993,
     'simply': 56865,
     'horrible': 30874,
     'along': 5184,
     'vanishing': 66397,
     'gone': 27575,
     '60': 3279,
     'seconds': 55041,
     'represent': 52025,
     'quintessential': 50212,
     'car': 12005,
     'films': 24554,
     'original': 44901,
     'stands': 59201,
     'head': 29505,
     'whereas': 68271,
     'gave': 26677,
     '75': 3387,
     '%': 3,
     '25': 2878,
     'drama': 20262,
     'giss': 27215,
     '2000': 2775,
     'cage': 11567,
     'super': 60672,
     'saves': 54259,
     'kisses': 35245,
     'girl': 27176,
     'mtv': 41774,
     'edits': 21188,
     'tits': 63139,
     'ass': 6761,
     'save': 54254,
     'rent': 51912,
     'did': 18798,
     'live': 37213,
     'baby': 7525,
     'wrote': 69343,
     'produced': 49169,
     'directed': 19070,
     'starred': 59275,
     "'74": 60,
     "'": 5,
     'degraded': 17982,
     'manner': 38698,
     'through': 62793,
     'title': 63130,
     'character': 12868,
     'glenn': 27302,
     'ford': 25367,
     'says': 54306,
     'fight': 24400,
     'forced': 25359,
     'intelligent': 32959,
     'strong': 60040,
     'beliefs': 8661,
     'courage': 16093,
     'striving': 60021,
     'feel': 24115,
     'mostly': 41458,
     'paced': 45596,
     'brian': 10651,
     'keith': 34885,
     'edward': 21212,
     'g.': 26316,
     'robinson': 52994,
     'image': 31796,
     'thoroughly': 62662,
     'enjoyable': 22032,
     'includes': 32234,
     'strategic': 59885,
     'non-military': 43408,
     'warfare': 67536,
     'violent': 66960,
     'states': 59326,
     'graphic': 27953,
     'computer-generated': 14985,
     'era': 22314,
     'personality': 46849,
     'clashes': 13813,
     'happened': 29119,
     'upon': 66037,
     '8-10': 3412,
     'dark': 17348,
     'november': 43748,
     'afternoon': 4587,
     'outside': 45187,
     'day': 17496,
     'freezing': 25833,
     'around': 6529,
     '4pm': 3157,
     'cup': 16886,
     'hot': 30996,
     'cocoa': 14303,
     'sat': 54164,
     'front': 25994,
     'blanket': 9466,
     'cartoon': 12235,
     'happy': 29137,
     'silly': 56788,
     '--': 1696,
     'moralistic': 41308,
     'captured': 11998,
     'imagination': 31808,
     "'m": 911,
     'sure': 60880,
     'misses': 40771,
     'text': 62297,
     'abbreviated': 3710,
     'wrong': 69324,
     'places': 47533,
     'tolkien': 63254,
     'purist': 49896,
     'captures': 12000,
     'spirit': 58723,
     'choice': 13408,
     'carry': 12214,
     'burden': 11219,
     'consequences': 15339,
     'selfish': 55334,
     'rash': 50672,
     'decisions': 17759,
     'etc': 22539,
     'quality': 50069,
     'animation': 5715,
     'leaves': 36396,
     'complaint': 14881,
     'where': 68267,
     'rises': 52850,
     'above': 3808,
     'voice': 67116,
     'characterizations': 12900,
     'john': 34233,
     'hurt': 31360,
     'drawn': 20323,
     'listen': 37142,
     'extraordinary': 23345,
     'again': 4608,
     'many': 38755,
     'times': 63018,
     'brings': 10740,
     'back': 7562,
     'kid': 35051,
     'thirsty': 62609,
     'magical': 38276,
     'adventure': 4426,
     'reason': 51073,
     "'lucky": 909,
     'nostalgic': 43620,
     'overlook': 45415,
     'shortcomings': 56373,
     'between': 8985,
     'fantasy': 23766,
     'reached': 50950,
     'favourite': 24014,
     'much': 41776,
     'going': 27514,
     'things': 62566,
     'wonder': 68967,
     'kieslowski': 35088,
     'mean': 39534,
     'instance': 32859,
     'judge': 34404,
     'meant': 39564,
     'god': 27442,
     'supreme': 60870,
     'irene': 33395,
     'jacob': 33750,
     'double': 20093,
     'veronique': 66636,
     'outstanding': 45201,
     'few': 24300,
     'superficially': 60758,
     'prettier': 48930,
     'actresses': 4191,
     'none': 43493,
     'manage': 38578,
     'beauty': 8433,
     'does.': 19813,
     'tritingnant': 64175,
     'magnificent': 38300,
     'really': 51037,
     'saying': 54300,
     'answer': 5824,
     'valentine': 66302,
     'biggest': 9134,
     'advertising': 4451,
     'campaign': 11749,
     'russia': 53631,
     'epic': 22245,
     'russian': 53633,
     'culture': 16859,
     'saga': 53843,
     'endless': 21920,
     'articles': 6605,
     'interviews': 33165,
     'disappointment': 19188,
     'main': 38354,
     'played': 47637,
     'oleg': 44445,
     'menshikov': 39853,
     'stupid': 60179,
     'immature': 31859,
     'boy': 10363,
     'ready': 50985,
     'set': 55667,
     'comrades': 15001,
     'lady': 35804,
     'admire': 4335,
     'develop': 18586,
     'festival': 24267,
     'scenes': 54451,
     'boasting': 9790,
     'audacity.': 7085,
     'respect': 52213,
     'mikhalkov': 40300,
     'previous': 48959,
     'works': 69102,
     'both': 10245,
     'demonstrates': 18161,
     'ambitions': 5352,
     'tzar': 64769,
     'remember': 51807,
     'hitch': 30398,
     'hiking': 30222,
     'getting': 26998,
     'lift': 36896,
     'turned': 64493,
     'two': 64663,
     'fleeing': 24953,
     'italian': 33604,
     'crooks': 16601,
     'law': 36249,
     'side': 56639,
     'carrying': 12216,
     'little': 37196,
     'portrait': 48240,
     'jesus': 34081,
     'pocket': 47846,
     'protection': 49465,
     '...': 1927,
     'unjust': 65619,
     'criminal': 16490,
     'correct': 15871,
     'combination': 14586,
     'categories': 12413,
     'used': 66162,
     'know': 35386,
     "'les": 874,
     'valseuses': 66340,
     'gives': 27236,
     'although': 5251,
     'picture': 47230,
     'close': 14055,
     'experiences': 23153,
     'intenseness': 32973,
     'each': 20951,
     'soon': 58207,
     "'all": 119,
     'behind': 8610,
     'momentous': 41050,
     'state': 59311,
     'mind': 40445,
     "'dangerous": 393,
     'form': 25469,
     'willing': 68605,
     'persecute': 46813,
     "'stealing": 1430,
     'touches': 63567,
     "'values": 1575,
     "'what": 1618,
     'morals': 41318,
     'resurrection': 52312,
     'individual': 32407,
     'him/': 30270,
     'free': 25791,
     "'false": 535,
     'conformism': 15205,
     'danger': 17275,
     'far': 23780,
     "'freedom": 601,
     'crossing': 16642,
     'borders': 10156,
     'jail': 33782,
     'shoot': 56313,
     'stealing': 59386,
     'petrol': 46958,
     'multinational': 41892,
     'oil': 44364,
     'company': 14809,
     'fifth': 24387,
     'hand': 28992,
     'les': 36624,
     're-awakens': 50816,
     'questions': 50156,
     'morality': 41310,
     'contradiction': 15559,
     'usual': 66190,
     "'media": 952,
     'message': 39984,
     'complex': 14907,
     'bored': 10163,
     'decided': 17745,
     'called': 11652,
     "'out": 1079,
     'kold': 35451,
     'once': 44527,
     'bought': 10280,
     'whole': 68411,
     'cheap': 13050,
     'movies': 41682,
     'videostore': 66801,
     'seemed': 55133,
     'chips': 13373,
     'coke': 14398,
     'relaxing': 51691,
     'evening': 22659,
     'pain': 45677,
     'started': 59288,
     'beginning': 8574,
     'boxer': 10353,
     'nicest': 42979,
     'person': 46834,
     'nice': 42967,
     'pussy': 49965,
     'totally': 63544,
     'fit': 24766,
     '28': 2907,
     'ko': 35421,
     'starts': 59299,
     'pimp': 47342,
     'earn': 20988,
     'extra': 23322,
     'guys': 28609,
     'every': 22727,
     'punch': 49802,
     'missed': 40768,
     'annoying': 5796,
     'effect': 21233,
     'synchronised': 61330,
     'punches': 49807,
     ...}




```python
# These will now contain for each row a list of unique integer number and
# its associated importance as calculated by TF-IDF.
train_x_tfidf = tfidf.transform(train_x)
test_x_tfidf = tfidf.transform(test_x)
print("Word vectorization is complete.")
print("\n- Tfidf vectorized format of training set: ")
print(train_x_tfidf)
print("\n- Tfidf vectorized format of testing set: ")
print(test_x_tfidf)
```

    Word vectorization is complete.
    
    - Tfidf vectorized format of training set: 
      (0, 69319)	0.1333107841352205
      (0, 68453)	0.09686396946216957
      (0, 63168)	0.035648091266605106
      (0, 62619)	0.07401428821871625
      (0, 62412)	0.0846022639740091
      (0, 62401)	0.07704150198937129
      (0, 62372)	0.13549840253129092
      (0, 62349)	0.0813873561252322
      (0, 58114)	0.11955886922256159
      (0, 55667)	0.12097732285699019
      (0, 53974)	0.10122078772262247
      (0, 53123)	0.11353527279670876
      (0, 48119)	0.16197209406167434
      (0, 47659)	0.13115898019097957
      (0, 44548)	0.053590604667238015
      (0, 44495)	0.09887199604662286
      (0, 43670)	0.15633638221052534
      (0, 43208)	0.07274894613654179
      (0, 42332)	0.3540066319175665
      (0, 41682)	0.08578826352200217
      (0, 38198)	0.08154143452626555
      (0, 36950)	0.05964139561831107
      (0, 36368)	0.2038288863339496
      (0, 33486)	0.07430494156059743
      (0, 33035)	0.15752452831164587
      :	:
      (7999, 29419)	0.10627047336827755
      (7999, 29164)	0.1202551183387072
      (7999, 27592)	0.20593045376598176
      (7999, 26112)	0.1342956689073716
      (7999, 24454)	0.11073175021293705
      (7999, 22915)	0.12480053684656774
      (7999, 22799)	0.3285961466056577
      (7999, 21470)	0.15975088276703778
      (7999, 21233)	0.1630156330768652
      (7999, 18586)	0.2021072157520159
      (7999, 16234)	0.2667334671966799
      (7999, 12314)	0.1070788632867638
      (7999, 11365)	0.04638862327123825
      (7999, 8962)	0.09488583231555993
      (7999, 8021)	0.17420736738755285
      (7999, 5814)	0.10170423440556024
      (7999, 5568)	0.03571326224029742
      (7999, 5018)	0.057676313905039675
      (7999, 4095)	0.08871015140862391
      (7999, 3612)	0.1426245639539534
      (7999, 3431)	0.17029455640628396
      (7999, 2983)	0.17064867457482044
      (7999, 1927)	0.09044447676220017
      (7999, 1924)	0.10511241007890085
      (7999, 1679)	0.14332549651435114
    
    - Tfidf vectorized format of testing set: 
      (0, 68955)	0.0726419529461905
      (0, 68823)	0.025187025795761417
      (0, 68584)	0.043303475077101465
      (0, 68411)	0.059538904314428646
      (0, 68397)	0.03275430897664936
      (0, 66960)	0.09324295087361684
      (0, 66164)	0.1185582599890449
      (0, 64485)	0.0728246623231353
      (0, 64303)	0.06806916046925839
      (0, 63621)	0.10341644690723735
      (0, 63168)	0.057871926385170205
      (0, 62619)	0.04005216256143033
      (0, 62603)	0.15024444843121662
      (0, 62401)	0.08338062382054416
      (0, 62372)	0.12831653600019113
      (0, 61865)	0.08598509417354239
      (0, 61514)	0.05701657439243712
      (0, 60797)	0.10713051287896694
      (0, 60466)	0.09967876988905726
      (0, 60127)	0.1026427373768479
      (0, 60110)	0.09935982434017519
      (0, 59420)	0.16041794446483718
      (0, 57547)	0.17301896510108053
      (0, 57350)	0.10918107066707271
      (0, 56892)	0.06137978443035679
      :	:
      (1999, 6391)	0.02183158862235434
      (1999, 6068)	0.028882915482061365
      (1999, 5814)	0.020308481601832417
      (1999, 5568)	0.07131287437417108
      (1999, 5502)	0.0236569145669787
      (1999, 5218)	0.032358732008992426
      (1999, 5182)	0.029948459310565643
      (1999, 5018)	0.04606763392490461
      (1999, 4608)	0.021292087218679936
      (1999, 4194)	0.031496640346680566
      (1999, 3800)	0.02602330157902497
      (1999, 3612)	0.07119881364717988
      (1999, 3574)	0.030627306456144074
      (1999, 3573)	0.08517495743941458
      (1999, 3564)	0.0852700834580908
      (1999, 2755)	0.0329319215995556
      (1999, 2024)	0.08520347802194594
      (1999, 1928)	0.02968031553853098
      (1999, 1924)	0.0839561286189641
      (1999, 1679)	0.07154872228397763
      (1999, 1672)	0.0113176386413923
      (1999, 1671)	0.011380780527464551
      (1999, 1582)	0.0395318616792649
      (1999, 1282)	0.037036302416848674
      (1999, 0)	0.014265944205288119
    


```python

# --- STEP 8: USE ML ALGORITHMS TO PREDICT OUTCOME ---
print("\n********** ML ALGORITHMS **********\n")
print("- Classifier 1: Multinomial Naive Bayes")
# fit the training dataset on the NB classifier
print("|___ Fitting training dataset on the MNB classifier...")
model_MNB = naive_bayes.MultinomialNB()
start = timeit.default_timer()
model_MNB.fit(train_x_tfidf, train_y)
stop = timeit.default_timer()
print("|___ Training is complete. Total time: ", stop - start)
# predict the labels on validation dataset
print("|___ Making predictions...")
start = timeit.default_timer()
predictions_MNB = model_MNB.predict(test_x_tfidf)
stop = timeit.default_timer()
print("|___ Finished making predictions. Total time: ", stop - start)
# Use accuracy_score function to get the accuracy
# ground truth: news_test.target; predicted labels: predicted
print("|___ Naive Bayes Accuracy Score -> ", accuracy_score(test_y, predictions_MNB)*100)
print(classification_report(test_y, predictions_MNB))
print("Confusion Matrix:\n", confusion_matrix(test_y, predictions_MNB))
```

    
    ********** ML ALGORITHMS **********
    
    - Classifier 1: Multinomial Naive Bayes
    |___ Fitting training dataset on the MNB classifier...
    |___ Training is complete. Total time:  0.02125669999998081
    |___ Making predictions...
    |___ Finished making predictions. Total time:  0.003428499999927226
    |___ Naive Bayes Accuracy Score ->  84.7
                  precision    recall  f1-score   support
    
               0       0.80      0.93      0.86      1006
               1       0.91      0.77      0.83       994
    
        accuracy                           0.85      2000
       macro avg       0.86      0.85      0.85      2000
    weighted avg       0.86      0.85      0.85      2000
    
    Confusion Matrix:
     [[931  75]
     [231 763]]
    


```python

# Classifier 2 - Algorithm - SVM
# fit the training dataset on the SVM classifier
print("\n- Classifier 2: Support Vector Machines (SVM)")
print("|___ Fitting training dataset on the SVM classifier...")
model_SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
start = timeit.default_timer()
model_SVM.fit(train_x_tfidf, train_y)
stop = timeit.default_timer()
print("|___ Training is complete. Total time: ", stop-start)
# predict the labels on validation dataset
print("|___ Making predictions...")
start = timeit.default_timer()
predictions_SVM = model_SVM.predict(test_x_tfidf)
stop = timeit.default_timer()
print("|___ Finished making predictions. Total time: ", stop-start)
# Use accuracy_score function to get the accuracy
print("|___ SVM Accuracy Score -> ", accuracy_score(test_y, predictions_SVM)*100)
print(classification_report(test_y, predictions_SVM))
print("Confusion Matrix:\n", confusion_matrix(test_y, predictions_SVM))
```

    
    - Classifier 2: Support Vector Machines (SVM)
    |___ Fitting training dataset on the SVM classifier...
    |___ Training is complete. Total time:  42.10728549999999
    |___ Making predictions...
    |___ Finished making predictions. Total time:  10.662308800000005
    |___ SVM Accuracy Score ->  89.55
                  precision    recall  f1-score   support
    
               0       0.89      0.90      0.90      1006
               1       0.90      0.89      0.89       994
    
        accuracy                           0.90      2000
       macro avg       0.90      0.90      0.90      2000
    weighted avg       0.90      0.90      0.90      2000
    
    Confusion Matrix:
     [[905 101]
     [108 886]]
    


```python

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    model_LR = LogisticRegression(C=c, max_iter=1000)
    # fit the training dataset on the Logistic Regression classifier
    print("\nFitting training dataset on the Logistic Regression classifier...")
    start = timeit.default_timer()
    model_LR.fit(train_x_tfidf, train_y)
    stop = timeit.default_timer()
    print("Training is complete. Total time: ", stop - start)
    # predict the labels on validation dataset (i.e. test dataset)
    print("Making predictions...")
    start = timeit.default_timer()
    predictions_LR = model_LR.predict(test_x_tfidf)
    stop = timeit.default_timer()
    print("- Finished making predictions! Total Time: ", stop - start)
    print("- LR Accuracy Score for C=%s: %s" % (c, accuracy_score(test_y, predictions_LR)*100))
    print(classification_report(test_y, predictions_LR))
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions_LR))
```

    
    Fitting training dataset on the Logistic Regression classifier...
    Training is complete. Total time:  0.29214919999992617
    Making predictions...
    - Finished making predictions! Total Time:  0.0013181000000486165
    - LR Accuracy Score for C=0.01: 74.6
                  precision    recall  f1-score   support
    
               0       0.77      0.71      0.74      1006
               1       0.73      0.78      0.75       994
    
        accuracy                           0.75      2000
       macro avg       0.75      0.75      0.75      2000
    weighted avg       0.75      0.75      0.75      2000
    
    Confusion Matrix:
     [[712 294]
     [214 780]]
    
    Fitting training dataset on the Logistic Regression classifier...
    Training is complete. Total time:  0.231114600000069
    Making predictions...
    - Finished making predictions! Total Time:  0.001548399999933281
    - LR Accuracy Score for C=0.05: 78.55
                  precision    recall  f1-score   support
    
               0       0.80      0.76      0.78      1006
               1       0.77      0.81      0.79       994
    
        accuracy                           0.79      2000
       macro avg       0.79      0.79      0.79      2000
    weighted avg       0.79      0.79      0.79      2000
    
    Confusion Matrix:
     [[762 244]
     [185 809]]
    
    Fitting training dataset on the Logistic Regression classifier...
    Training is complete. Total time:  0.43850359999999
    Making predictions...
    - Finished making predictions! Total Time:  0.0010290999999824635
    - LR Accuracy Score for C=0.25: 85.25
                  precision    recall  f1-score   support
    
               0       0.87      0.84      0.85      1006
               1       0.84      0.87      0.85       994
    
        accuracy                           0.85      2000
       macro avg       0.85      0.85      0.85      2000
    weighted avg       0.85      0.85      0.85      2000
    
    Confusion Matrix:
     [[841 165]
     [130 864]]
    
    Fitting training dataset on the Logistic Regression classifier...
    Training is complete. Total time:  0.4346676000000116
    Making predictions...
    - Finished making predictions! Total Time:  0.0012991999999485415
    - LR Accuracy Score for C=0.5: 87.25
                  precision    recall  f1-score   support
    
               0       0.88      0.87      0.87      1006
               1       0.87      0.88      0.87       994
    
        accuracy                           0.87      2000
       macro avg       0.87      0.87      0.87      2000
    weighted avg       0.87      0.87      0.87      2000
    
    Confusion Matrix:
     [[873 133]
     [122 872]]
    
    Fitting training dataset on the Logistic Regression classifier...
    Training is complete. Total time:  0.5891338000000133
    Making predictions...
    - Finished making predictions! Total Time:  0.0013218999999935477
    - LR Accuracy Score for C=1: 88.64999999999999
                  precision    recall  f1-score   support
    
               0       0.89      0.89      0.89      1006
               1       0.88      0.89      0.89       994
    
        accuracy                           0.89      2000
       macro avg       0.89      0.89      0.89      2000
    weighted avg       0.89      0.89      0.89      2000
    
    Confusion Matrix:
     [[891 115]
     [112 882]]
    


```python

```
