import pandas as pd
import bert.helpers as h

tr = h.train_data()
te = h.test_data()
st = h.stance_data()

st["controversial trending issue"].value_counts()
st["stance"] = "NONE"
st.to_csv(h.dataset_path + "stance_predict.csv")

test_set = pd.concat([
    st[st["controversial trending issue"] == topic].sample(10)
    for topic in h.TOPICS
    if topic is not None
])
test_set.to_csv(h.dataset_path + "stance_test_random.csv")
