#!/usr/bin/env python
# coding: utf-8

# # Data Preparation
# 
# ## a) Read the selected data, list the fields/variables, and identify their types.

# In[124]:


#Reading and displaying the data 
import pandas as pd
df=pd.read_csv("Churn_Modelling2.csv")

print("\n")
display(df.info())
print("\n"*2)

#listing the columns according to their type
NUMERICAL_columns = df.select_dtypes(exclude='object').columns.tolist()
CATEGORICAL_columns = df.select_dtypes(include='object').columns.tolist()


print('Numerical columns are:',NUMERICAL_columns)
print("\n")
print('CATEGORICAl columns are:', CATEGORICAL_columns)
print("\n"*2)
#desceibe the columns according to their type
print(f' the Numerical columns are \n:{df[NUMERICAL_columns].describe()}\n' )
print(f' the CATEGORICAl columns are \n:{df[CATEGORICAL_columns].describe()}\n' )
# number of the row and columns
print(f'The number of rows: {len(df.index)}, the number of columns : {len(df.columns)}.')
print(f'The number of non-null rows for each column are:\n{df.count()}')

display(df.head(10))







# ## b)  List the missing data, and outliers. Fix the inconsistencies, impute the missing data and remove the outliers.

# In[125]:


#  Identify the columns containing null values
#null_columns=df.columns[df.isna().any()] thre are not null so we will adjust data
#print('Columns with NaN values are:', null_columns)
#drop the fist column that contain row number 
import pandas as pd

# Assuming you have already loaded your DataFrame as df

# Drop the 'RowNumber' column
# df.to_csv('Churn_Modelling2-Clean.csv', index = False, header=True)
df = df.drop(["RowNumber"], axis=1)

#df.to_csv('Churn_Modelling2.csv',index=False)
# Modify the data
df['HasCrCard'] = df['HasCrCard'].apply(lambda x: "No" if x == 0 else "Yes")
df['IsActiveMember'] = df['IsActiveMember'].apply(lambda x: "No" if x == 0 else "Yes")
df['Exited'] = df['Exited'].apply(lambda x: "No" if x == 0 else "Yes")
#fill messing age
median_age = df['Age'].median()
rounded_median_age = round(median_age)
df['Age'].fillna(rounded_median_age, inplace=True)


#fill messing geography
geography_mode = df['Geography'].mode()[0]
df['Geography'].fillna(geography_mode, inplace=True)

#number of products missing
mean_num_of_products = df['NumOfProducts'].mean()
rounded_mean = round(mean_num_of_products)
df['NumOfProducts'].fillna(rounded_mean, inplace=True)

display(df.head(55))


# In[126]:


#remove outliers
import seaborn as sns
import matplotlib.pyplot as plt 
print(df.info())
Numerical_columns=df.select_dtypes(exclude='object').columns.tolist()
CATEGORICAL_columns=df.select_dtypes(include='object').columns.tolist()
print(Numerical_columns)
print(CATEGORICAL_columns)
Numerical_columns.remove("CustomerId")
print('Numerical columns are:',CATEGORICAL_columns)
for i in Numerical_columns:
    plt.figure()
    sns.boxplot(y=i,x='HasCrCard',data=df);
    plt.show()

def remove_outliers(df):
    selected_cols= df.select_dtypes(exclude='object').columns
    for col in selected_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1


        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df

df=remove_outliers(df)


# In[127]:


# after removing the outliers
for i in Numerical_columns:
    plt.figure()
    sns.boxplot(y=i,x='HasCrCard',data=df);
    plt.show()


# # Model Planning:

# ## a. Run the exploratory data analysis:
# 

# ### i) Find the statistical summaries
# 

# In[128]:


# Identify the data type of each column
df.info()

# Give a statistical summary of the columns
print("\n"*2)
print("#"*32)
print("#The summry of Categorical Data#")
print("#"*32)
display(df[CATEGORICAL_columns].describe())

print("\n"*2)
print("#"*30)
print("#The summry of Numerical Data#")
print("#"*30)
display(df.describe())
print("#"*30)
target_Balance = round(df['Balance'].quantile(0.4),3)
print(f'The salary of targeted customers starts from {target_Balance}')


# ### ii. Make univariate graphs (i.e., graphs based on single variable)

# In[129]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.xticks(rotation=45)
plt.hist(data=df,x="Balance",bins=10)
plt.title('Balance')
plt.show()
subj_labels, subj_counts = np.unique(df["Geography"],return_counts=True)
plt.figure()
plt.pie(subj_counts, labels = subj_labels,autopct='%.2f%%') #autopct to format labels
plt.show()
Geography=df["Geography"].unique()


# ### iii. Prepare bivariate plots (i.e., plots based on two variables).

# In[130]:


display(df)
plt.figure(figsize=(15,8))
sns.scatterplot(x='NumOfProducts', y='Balance', 
                size='CreditScore',sizes=(20,200),
                alpha=0.5, color='red', 
                data=df)
plt.xticks(rotation=90)
plt.show()
print("\n"*5)
plt.figure(figsize=(15,8))
sns.boxplot(x ='Geography', y ='Balance', data = df) 
plt.xticks(rotation=90)
plt.show()



# In[ ]:





# ### iv. Portray advanced graphs (i.e., graphs based on more than two variables).

# In[131]:


plt.figure(figsize=(15,8))
sns.lineplot(x ='IsActiveMember', y ='EstimatedSalary', hue="Geography",markers=True,data = df) 
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(15, 8))

print("\n"*5)
plt.figure(figsize=(15,8))
sns.violinplot(x ='Geography', y ='Balance',hue="Gender", data = df) 
plt.xticks(rotation=90)

plt.figure()
sns.pairplot(vars=Numerical_columns,hue ='Gender',data=df)   # look for {x, y}_vars kwargs
plt.show()

plt.figure(figsize=(5,5))
sns.histplot(x='Age', y='Balance', bins=10, data=df)
plt.show()


# ### v. Assess the relationship between variables.

# In[132]:


from sklearn.decomposition import PCA




#df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
#plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",linewidths=.5)
# plt.title('Correlation Matrix')
# plt.show()


# ## b. Summarise your findings. 

# In[133]:


df


# # 4. Model Building:

# ## a. Estimate the unknown model parameters (fitting) and evaluate the model
# (validation/cross-validation).

# In[134]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[["Age","Tenure","NumOfProducts","CreditScore","EstimatedSalary"]]
y = df['Balance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression() # Train the model
model.fit(X_train, y_train)
# Scaling the Train - Test splits
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(np.c_[X_train,y_train])

A_train = scaler.transform(np.c_[X_train,y_train])
X_train = A_train[:,:-1]
y_train = A_train[:,-1]

A_test = scaler.transform(np.c_[X_test,y_test])
X_test = A_test[:,:-1]
y_test = A_test[:,-1]


# In[135]:


from sklearn.metrics import mean_squared_error

## OLS
from sklearn.linear_model import LinearRegression
reg1 = LinearRegression(fit_intercept=False).fit(X_train, y_train)
y_pred1 = reg1.predict(X_test)
print('The best coefficient estimates are:', reg1.coef_)
print('The MSE using OLS is:', mean_squared_error(y_test, y_pred1))


## Ridge
from sklearn.linear_model import RidgeCV
reg2 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], fit_intercept=False,cv=10).fit(X_train, y_train)
y_pred2 = reg2.predict(X_test)
print('The MSE using Ridge is:', mean_squared_error(y_test, y_pred2))


## Lasso
from sklearn.linear_model import LassoCV
reg3 = LassoCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], 
               fit_intercept=False,cv=10, random_state=0).fit(X_train, y_train)
y_pred3 = reg3.predict(X_test)
print('The MSE using Lasso is:', mean_squared_error(y_test, y_pred3))


# In[136]:


def Geography(x):
    if x=="France":
        return 0
    elif x=="Spain":
        return 1
    else:
        return 2
df["Geography"]=df["Geography"].apply(Geography)
df.loc[:,["NumOfProducts","Age","EstimatedSalary"]]=df.loc[:,["NumOfProducts","Age","EstimatedSalary"]].applymap(lambda x: int(round(x)))
df.loc[:,['HasCrCard','IsActiveMember','Exited']]=df.loc[:,['HasCrCard','IsActiveMember','Exited']].applymap(lambda x: 0 if x=="No" else 1)
df.loc[:,"Gender"]=df.loc[:,"Gender"].apply(lambda x: 0 if x=="Female" else 1)
df.loc[:,"Balance"]=df.loc[:,"Balance"].apply(lambda x: 1 if x>=target_Balance else 0)
display(df.head())
X = df.drop(["CustomerId","Surname","Balance"],axis=1).values
y = df['Balance'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[137]:


from sklearn import tree
dtClf = tree.DecisionTreeClassifier(random_state=0,criterion='entropy',splitter='best')
dtClf = dtClf.fit(X_train,y_train)
dt_y_pred = dtClf.predict(X_test)
c=df.columns.tolist()
c.remove("Balance")
import matplotlib.pyplot as plt
plt.figure(figsize =(15,15),dpi=1000)
tree.plot_tree(dtClf,feature_names=c[2:],class_names=['0','1'],filled=True,rounded=True,max_depth=3); 
plt.show()


# In[138]:


from sklearn.metrics import accuracy_score, confusion_matrix
print("Decision Tree: \n")
print("Accuracy:=",  accuracy_score(y_test, dt_y_pred))
print("Confusion Matrix:= \n", confusion_matrix(y_test, dt_y_pred) )

from sklearn.naive_bayes import GaussianNB
NBClf2 = GaussianNB()   
NBClf2.fit(X_train,y_train)
NB_y_pred = NBClf2.predict(X_test)

print("\nNB for Numerical  Data: \n")
print("Accuracy:=",  accuracy_score(y_test, NB_y_pred))
print("Confusion Matrix:= \n", confusion_matrix(y_test, NB_y_pred) )

