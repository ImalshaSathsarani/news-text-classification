import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def evaluate_models_with_cv(pipelines,param_grids, X_train,X_test, y_train,y_test,target_names):
    results =[]
    best_model =None
    best_score = 0
    best_model_name =""

    print("\nTraining Models with Cross-Validation & Hyperparameter Tuning...\n")

    for name, pipeline in pipelines.items():
        print(f" Tuning {name}...")

        grid = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        preds = best_estimator.predict(X_test)
        acc = accuracy_score(y_test, preds)



        results.append({
            'Model':name,
            'CV Best Score':grid.best_score_, 
            'Test Accuracy':acc})
        

        print(f"Best CV Score: {grid.best_score_:.4f}")
        print(f"Test Accuracy: {acc:.4f}\n")

        if acc > best_score:
            best_score = acc
            best_model = best_estimator
            best_model_name = name
        
    results_df = pd.DataFrame(results)

    print("\nModel Comparison:")
    print(results_df)

    print(f"\nDetailed Report for Best Model: {best_model_name}")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=target_names))

    return best_model, best_model_name, results_df