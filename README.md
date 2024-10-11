The first method that I am thinking about is the devision tree which is because the character is easy to be find 

Here is a record of method that I used in this competiton:
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
# ANN
