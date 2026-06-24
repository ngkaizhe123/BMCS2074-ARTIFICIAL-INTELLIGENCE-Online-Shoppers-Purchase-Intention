# app.py
import streamlit as st
import pandas as pd
import joblib  # 用于加载训练好的模型（或者直接从你的 model 脚本导入）

# 1. 设置网页标题和侧边栏
st.title("🛍️ 电子商务用户购买意向预测系统")
st.sidebar.header("选择机器学习模型")
model_choice = st.sidebar.selectbox(
    "想要使用哪个算法进行预测？", ["ANN (神经网络)", "SVM (支持向量机)", "KNN (最近邻)"]
)

# 2. 收集用户在网页上的输入
st.subheader("输入用户行为数据")
col1, col2 = st.columns(2)

with col1:
    administrative = st.number_input("浏览管理页面次数", min_value=0, value=0)
    informational_duration = st.number_input(
        "浏览信息页面总时间 (秒)", min_value=0.0, value=0.0
    )

with col2:
    exit_rates = st.slider("该页面的退出率 (Exit Rate)", 0.0, 1.0, 0.02)
    is_weekend = st.selectbox("是否是周末？", ["否", "是"])

# 3. 当用户点击“开始预测”按钮时
if st.button("开始预测"):
    # 3.1 把输入的预测数据打包（包装成和训练时一样的特征格式）
    user_data = {
        "Administrative": administrative,
        "Informational_Duration": informational_duration,
        "ExitRates": exit_rates,
        "Weekend": 1 if is_weekend == "是" else 0,
    }
    input_df = pd.DataFrame([user_data])

    # 【注意】这里需要调用你在 src/data_preprocessing.py 里写好的标准化逻辑
    # scaled_data = your_scaler.transform(input_df)

    # 3.2 根据用户的选择，调用对应的模型
    if model_choice == "ANN (神经网络)":
        # y_pred = ann_model.predict(scaled_data)
        st.write("正在使用 ANN 模型计算...")
    elif model_choice == "SVM (支持向量机)":
        # y_pred = svm_model.predict(scaled_data)
        st.write("正在使用 SVM 模型计算...")

    # 3.3 把预测结果漂亮地展示出来
    # list_result = 1 代表会买，0 代表不会买
    y_pred = 1  # 假设预测结果
    if y_pred == 1:
        st.success("🎉 预测结果：该用户**极大概率会购买**！可以向其推送优惠券。")
    else:
        st.warning("💤 预测结果：该用户可能只是随便逛逛，**购买意向较低**。")
