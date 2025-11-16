"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""
import numpy as np
from catboost import CatBoostClassifier
import os
import pandas as pd

def create_submission(predictions, test_df):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    
    submission = pd.DataFrame({
        "id": test_df["id"],
        "p_mens_email": predictions[:, 0],
        "p_womens_email": predictions[:, 1],
        "p_no_email": predictions[:, 2]
        })
    
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path



def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    

    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    df = pd.read_csv('/app/data/train.csv')

    ACTION_LIST = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
    ACTION2ID = {a: i for i, a in enumerate(ACTION_LIST)}
    df["action_id"] = df["segment"].map(ACTION2ID).astype(int)

    features = ["recency", "history", "mens", "womens", "newbie",
            "zip_code", "channel", "history_segment"]

    X = df[features].copy()
    X["action_id"] = df["action_id"]
    y = df["visit"]

    cat_features = ["zip_code", "channel", "history_segment", "action_id"]

    model = CatBoostClassifier(
    iterations=700,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=100
    )

    model.fit(X, y, cat_features=cat_features)

    def get_p_visit(x_row):
        p = []
        for aid in range(3):
            x_temp = x_row.copy()
            x_temp["action_id"] = aid
            pk = model.predict_proba(x_temp.values.reshape(1, -1))[0, 1]
            p.append(pk)
        return np.array(p)

    def soft_policy(p, T=90):
        p = p ** T
        return p / p.sum()


    test_df = pd.read_csv(r'/app/data/test.csv')
    Xtest = test_df[features].copy()

    predictions = []

    for i in range(len(Xtest)):
        p = get_p_visit(Xtest.iloc[i])
        pi = soft_policy(p, T=90)
        predictions.append(pi)

    predictions = np.array(predictions)
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(predictions, test_df)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
