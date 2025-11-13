# -----------------------------
# Example 1: Logistic Regression - Pass/Fail based on Hours
# -----------------------------
data = {
    "Hours": [1,3,5,8,10],
    "Marks": [0,0,1,1,1]
}
df = pd.DataFrame(data)
X = df[["Hours"]]
y = df["Marks"]

log_model = LogisticRegression()
log_model.fit(X, y)

y_pred = log_model.predict(X)
y_prob = log_model.predict_proba(X)[:, 1]

print("Predicted Classes: ", y_pred)
print("Predicted Probabilities: ", y_prob)

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

print("Accuracy: ", acc)
print("Precision: ", prec)
print("Recall: ", rec)
print("Confusion Matrix:\n", cm)

# -----------------------------
# Example 2: Logistic Regression - Train-Test Split
# -----------------------------
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

test_X = np.array([[3.5], [7.5]])
y_class_pred = log_model.predict(test_X)
y_prob_pred = log_model.predict_proba(test_X)

print(f"Predicted Classes: {y_class_pred}")
print(f"Predicted Probabilities:\n{y_prob_pred}")

# -----------------------------
# Example 3: Spam Classifier
# -----------------------------
X_spam = np.array([[10, 1], [5, 0], [100, 10], [15, 2], [200, 30], [2, 0], [40, 5], [12, 1]])
y_spam = np.array([0, 0, 1, 0, 1, 0, 1, 0])

X_train, X_test, y_train, y_test = train_test_split(X_spam, y_spam, test_size=0.25, random_state=42, stratify=y_spam)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_class = log_model.predict(X_test)
y_pred_proba = log_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class, zero_division=0)
recall = recall_score(y_test, y_pred_class, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred_class)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, AUC: {auc_score:.2f}")
print("Confusion Matrix:\n", conf_matrix)
