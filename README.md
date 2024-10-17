The first method that I am thinking about is the devision tree which is because the character is easy to be find 

Here is a record of method that I used in this competiton:

# basic setting for package and instructment:
Python import package: 
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import ollama

```

Setting necessary package setting: 
```
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# This part is to solve the Chinese character problems
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Make sure that the Sheet can be viewed entirely
```
# Load data file 
```
wk = pd.read_csv(r"C:\Users\jshen67\Downloads\examples.csv")

wk_copy = wk.copy()
```

# Dealt with data
This is how we dealing with data:
```
def data_clean(df):
    df_copy = df.copy()
    df_copy.drop(columns=['q21','q22'], axis=0, inplace=True)
# we delete the part 'q21' and 'q22', this is the second part of this problem. 
    df_copy = pd.get_dummies(df_copy, columns=['q1','q2'])
    df_copy = df_copy.astype(int)
# Here we do the one-hot encoding. 
    print(df_copy.head())
    return df_copy

def test_train_data(df_train):
    y = df_train['label']
# Then let's do the one-hot encoding for label, y 
    X = df_train.drop(columns=['label'], axis=1)
    y_adjusted = y - 1
    y = to_categorical(y_adjusted, num_classes=6)
# Seperate the data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test
```
Here in this project what we are doing is we first use the **one-hot encoding method** to achieve the problem that in the questionaire that:

Please share all of the reasons you chose to attend the TAPS [SEMINAR NAME]
0. I had attended a previous Regional or Nationals seminar
1. To connect with other survivors who share a similar loss 
2. To learn more about resources TAPS has to offer 
3. To learn new tools and information to help me with my grief
4. To learn more about how to support my adult family members in their grief
5. To learn more about how to support my child(ren) in their grief 
6. For my child(ren) to attend Good Grief Camp
7. For my child(ren) to connect with a Military Mentor 

How did you find out about the TAPS [SEMINAR NAME]?
0. I found this event while searching for grief resources 
1. I attended a seminar last year and had already marked my calendar! 
2. TAPS invited me to this event via email 
3. My TAPS Survivor Care Team Member invited me
4. My Peer Mentor or another survivor invited me 
5. Through the TAPS website 
6. Through the TAPS magazine 
7. Through a TAPS Social Media Page

Then we find this is no nummerical relationships between each option.
# CarBoost:
```
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score


# 选择特征(X)和标签(y)
X = df.iloc[:, 2:21]  # 选择第2列到第20列作为特征
y = df['label']  # 选择'label'列作为标签

# 将特征分为数值型和分类型
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# 设置K折交叉验证参数
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)

# 存储每次折叠的准确率
fold_accuracies = []

# 进行K折交叉验证
for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # 创建CatBoost分类器
    model = CatBoostClassifier(
        iterations=100, 
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        cat_features=cat_features,
        random_state=42
    )
    
    # 训练模型
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    
    # 在验证集上进行预测
    predictions = model.predict(X_val)
    
    # 计算准确率
    accuracy = accuracy_score(y_val, predictions)
    fold_accuracies.append(accuracy)
    
    print(f"Fold {fold} - Accuracy: {accuracy:.4f}")

# 计算平均准确率
avg_accuracy = np.mean(fold_accuracies)
print(f"\nAverage Accuracy: {avg_accuracy:.4f}")

# 在整个数据集上训练最终模型
final_model = CatBoostClassifier(
    iterations=100, 
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    cat_features=cat_features,
    # random_state=42
)

final_model.fit(X, y, verbose=0)

print("\nFinal model trained on entire dataset.")`
```
```
Result = 
```
# ANN - Transfomer
```
df_copy = data_clean(df)
X_train, X_test, y_train, y_test = test_train_data(df_copy)

# 准备K-fold交叉验证
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 存储每次fold的训练历史
histories = []

# 进行K-fold交叉验证
for fold, (train_index, val_index) in enumerate(kfold.split(X_train)):
    print(f"Fold {fold + 1}/{n_splits}")
    
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    model = create_model_transformer((X_train.shape[1], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    history = model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=10, verbose=1, callbacks=[early_stopping])
    
    histories.append(history)

# 可视化结果
plt.figure(figsize=(12, 4))

# 绘制训练和验证准确率
plt.subplot(121)
for i, history in enumerate(histories):
    plt.plot(history.history['accuracy'], label=f'Train (Fold {i+1})')
    plt.plot(history.history['val_accuracy'], label=f'Validation (Fold {i+1})')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# 绘制训练和验证损失
plt.subplot(122)
for i, history in enumerate(histories):
    plt.plot(history.history['loss'], label=f'Train (Fold {i+1})')
    plt.plot(history.history['val_loss'], label=f'Validation (Fold {i+1})')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 计算并打印平均性能
val_accuracies = [history.history['val_accuracy'][-1] for history in histories]
mean_accuracy = np.mean(val_accuracies)
std_accuracy = np.std(val_accuracies)
print(f"Mean validation accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

# 在测试集上评估最终模型
final_model = create_model(X_train.shape[1])
final_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=5)], validation_split=0.2)
test_loss, test_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
```

Here we use two types of ANN, another good example is CNN



# The next part is called LLM with LLama3.1:

For each str question, we definne two functions: 
```
def comment_analysis_on_q21(comment,i):
    prompt = str(
        r"""People are in these 5 stages: 1. Overwhelmed, loss of purpose; shock and trauma emotions (isolation) present and challenging to understand. Individuals may struggle to deal with family responsibilities alone. Surviving Child: Feeling disconnected without guidance and attention from grieving adults.2. Experiencing tension between individuals within the family unit; lack of support from family members. Surviving Family Unit: Perception of other family members’ grief experience. Each family member may be at different phases of their grief journey.3. Experiencing grief and learning to process those emotions. Surviving Child: Seeks guidance and acknowledgment of grief; benefit from opportunities to open up and process with kids in similar situations to normalize emotions.4. Renewed experience of grief around anniversaries of loss, holidays, and special moments. Surviving Family Unit: Navigating special moments (sports, school achievements, moments that matter).5. Finding new purpose and goals to begin moving towards Positive Integration. Surviving Family Unit: Connected to a broader community; support system; not the only person/family experiencing loss.6. Healthy point in grief journey; feeling capable to help others and a desire to do so. Surviving Family Unit: Ready to give back to the TAPS community through mentorship programs, volunteering at charity drives & events, etc.According to their answer to this question, tell me their stage number:Please share with TAPS your favorite moment of the weekend? Did you have a breakthrough moment this weekend you would like to share? .{}Show me only the number"""
        ).format(comment)
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])['message']['content']
    return(response)    


def comment_analysis_on_q22(comment,i):
    prompt = str(
        r"""People are in these 5 stages: 1. Overwhelmed, loss of purpose; shock and trauma emotions (isolation) present and challenging to understand. Individuals may struggle to deal with family responsibilities alone. Surviving Child: Feeling disconnected without guidance and attention from grieving adults.2. Experiencing tension between individuals within the family unit; lack of support from family members. Surviving Family Unit: Perception of other family members’ grief experience. Each family member may be at different phases of their grief journey.3. Experiencing grief and learning to process those emotions. Surviving Child: Seeks guidance and acknowledgment of grief; benefit from opportunities to open up and process with kids in similar situations to normalize emotions.4. Renewed experience of grief around anniversaries of loss, holidays, and special moments. Surviving Family Unit: Navigating special moments (sports, school achievements, moments that matter).5. Finding new purpose and goals to begin moving towards Positive Integration. Surviving Family Unit: Connected to a broader community; support system; not the only person/family experiencing loss.6. Healthy point in grief journey; feeling capable to help others and a desire to do so. Surviving Family Unit: Ready to give back to the TAPS community through mentorship programs, volunteering at charity drives & events, etc.According to their answer to this question, tell me their stage number:Please share any additional feedback you have regarding your TAPS Seminar experience.{}Show me only the number"""
        ).format(comment)
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])['message']['content']
    return(response)
```


To use this prompt, we use 
```
# 假设 wk_copy 是已经定义的 DataFrame
wk_copy_q2122 = wk_copy[['q21', 'q22']]
print(wk_copy_q2122.head())
print(wk_copy_q2122.shape)

results = {}

for j in range(10):
    results[j] = pd.DataFrame(columns=['comment_q21', 'label_q21', 'comment_q22', 'label_q22'])
    comment_q21 = []
    label_q21 = []
    comment_q22 = []
    label_q22 = []

    for i in range(len(wk_copy_q2122)):
        print('------The {}---q21-------------'.format(i))
        label_21 = comment_analysis_on_q21(wk_copy_q2122.iloc[i, 0], i)
        print("The label is {}".format(label_21))
        print("the comment is {}".format(str(wk_copy_q2122.iloc[i, 0])))
        
        print('------The {}--q22-------------'.format(i))
        label_22 = comment_analysis_on_q22(wk_copy_q2122.iloc[i, 1], i)
        print("The label is {}".format(label_22))
        print("the comment is {}".format(str(wk_copy_q2122.iloc[i, 1])))
        print('-------------------------')

        comment_q21.append(wk_copy_q2122.iloc[i, 0])
        label_q21.append(label_21)
        comment_q22.append(wk_copy_q2122.iloc[i, 1])
        label_q22.append(label_22)

    results[j]['comment_q21'] = comment_q21
    results[j]['label_q21'] = label_q21
    results[j]['comment_q22'] = comment_q22
    results[j]['label_q22'] = label_q22

    # 保存表格
    results[j].to_csv(r"C:\Users\jshen67\Downloads\output_label_{}.csv".format(j), index=False)
    
```
The next thought is using the highest heat map to find the highest relationships between for each variables 
Only select few elements to use 

# Another idea to deal with is by collecting the number of each word, like "grief", "us" and using statistics to predict the stage 
Calcaulte the time that certain word coomes 
 Python 
```
def calculate_unique_word_frequencies(df):
    frequency = {label: Counter() for label in df['label'].unique()}
    help_count = 0  # count 

    for index, row in df.iterrows():
        comments = f"{row['comment_q21']} {row['comment_q22']}".lower()
        comments = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5\s]', '', comments)
        words = comments.split()
        
        # Get how many help exists 
        help_count += words.count('help')
        
        filtered_words = [word for word in words if word not in stop_words]
        frequency[row['label']].update(filtered_words)
    
    return frequency, help_count

# 计算所有不重复字词的频率
unique_frequencies, help_occurrences = calculate_unique_word_frequencies(df)


print(f"'help' 出现的次数: {help_occurrences}")

# 输出每个阶段的不重复词频统计
# for label, counts in unique_frequencies.items():
#     print(f"阶段 {label} 的不重复词频统计:")
#     for word, count in counts.items():
#         print(f"词 '{word}': {count} 次")
```
Or do a bar chart to check each word
```
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re

# 假设 df 是你的 DataFrame
# 计算某个词的频率
def calculate_all_word_frequencies(df):
    frequency = {label: Counter() for label in df['label'].unique()}
    
    for index, row in df.iterrows():
        comments = f"{row['comment_q21']} {row['comment_q22']}".lower()
        comments = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5\s]', '', comments)
        words = comments.split()
        frequency[row['label']].update(words)
    
    return frequency

# 计算所有字词的频率
all_frequencies = calculate_all_word_frequencies(df)

# 可视化
for label, counts in all_frequencies.items():
    words, counts = zip(*counts.most_common(10))  # 取前10个最常见的词
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.title(f"阶段 {label} 的词频统计")
    plt.xlabel("词")
    plt.ylabel("频率")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

```
