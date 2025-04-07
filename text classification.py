import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 读取数据文件（假设格式为：标签\t文本）
data = pd.read_csv('filtered_cnews.train.txt', sep='\t', header=None, names=['label', 'text'], quoting=3)

# 定义六个类别
categories = ["体育", "财经", "房产", "家居", "教育", "科技"]

# 初始化存储容器
X_train, X_test, X_val = [], [], []
y_train, y_test, y_val = [], [], []

# 按类别划分数据集
for category in categories:
    # 提取当前类别的数据
    category_data = data[data['label'] == category]
    texts = category_data['text'].values
    labels = category_data['label'].values

    # 打乱顺序并划分数据集
    texts_shuffled, labels_shuffled = shuffle(texts, labels, random_state=42)
    X_train.extend(texts_shuffled[:4000])
    X_test.extend(texts_shuffled[4000:4500])
    X_val.extend(texts_shuffled[4500:])

    y_train.extend(labels_shuffled[:4000])
    y_test.extend(labels_shuffled[4000:4500])
    y_val.extend(labels_shuffled[4500:])

# 转换为numpy数组
X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)

# 标签编码
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

# 文本向量化（TF-IDF）
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
X_val_vec = vectorizer.transform(X_val)

# 朴素贝叶斯模型
print("=" * 50 + "\n朴素贝叶斯模型训练及评估\n" + "=" * 50)
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

# 验证集评估
val_pred = nb.predict(X_val_vec)
print("验证集分类报告：")
print(classification_report(y_val, val_pred, target_names=le.classes_))

# 测试集评估
test_pred = nb.predict(X_test_vec)
print("测试集分类报告：")
print(classification_report(y_test, test_pred, target_names=le.classes_))

# KNN模型
print("\n" + "=" * 50 + "\nKNN模型训练及评估\n" + "=" * 50)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_vec, y_train)

# 验证集评估
val_pred_knn = knn.predict(X_val_vec)
print("验证集分类报告：")
print(classification_report(y_val, val_pred_knn, target_names=le.classes_))

# 测试集评估
test_pred_knn = knn.predict(X_test_vec)
print("测试集分类报告：")
print(classification_report(y_test, test_pred_knn, target_names=le.classes_))