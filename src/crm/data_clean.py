import numpy as np 
import pandas as pd 

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def re_format_data(df):

    # old column names
    campaigns_old = ['Cmp1','Cmp2','Cmp3','Cmp4','Cmp5']
    purchases_old = ['NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases' ] 
    products_old = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    
    # new column names
    campaigns =  ['Campaign 1','Campaign 2','Campaign 3','Campaign 4','Campaign 5']
    purchases = ['Deals Purchases', 'Web Purchases', 'Catalog Purchases', 'Store Purchases']
    products = ['Wine','Fruit','Meat','Fish','Sweet','Gold']

    column_mapping_campaign = {old_name: new_name for old_name, new_name in zip(campaigns_old, campaigns)}
    column_mapping_purchases = {old_name: new_name for old_name, new_name in zip(purchases_old, purchases)}
    column_mapping_product = {old_name: new_name for old_name, new_name in zip(products_old, products)}

    df = df.rename(columns=column_mapping_campaign)
    df = df.rename(columns=column_mapping_purchases)
    df = df.rename(columns=column_mapping_product)
     

    # Date related reformat
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
    df['Income'] = df['Income'].str.replace('$', '')
    df['Income'] = df['Income'].str.replace(',', '').astype(float)

    #Country Name format
    abbreviation_to_fullname = {
    'SP': 'Spain',
    'CA': 'Canada',
    'US': 'United States',
    'AUS': 'Australia',
    'GER': 'Germany',
    'IND': 'India',
    'SA': 'South Africa'
}
    df['Country'] = df['Country'].replace(abbreviation_to_fullname)
    
    # Number of days being customer
    df['Days Customers'] = (df['Dt_Customer'].max() - df['Dt_Customer']).dt.days
    
    # Age
    df['Age'] = df['Dt_Customer'].dt.year.max()-df['Year_Birth']
    
    # Total kids
    df['Total Kids']= df['Kidhome']+df['Teenhome']
    
    # Total Purchase
    df['Total Purchase'] = df[purchases].sum(axis=1)

    # Total Items
    df['Total Spend'] = df[products].sum(axis=1)

    # Total campaigns
    df['Total Campaigns'] = df[campaigns].sum(axis=1)

    # Housold as couples or singles
    df['Houshold Couple']= np.nan
    singles = (df['Marital_Status']=='Single' )| (df['Marital_Status']=='Widow' )| (df['Marital_Status']=='Divorced')|(df['Marital_Status']=='Alone')
    couples = (df['Marital_Status']=='Married' )| (df['Marital_Status']=='Together')
    df.loc[singles,'Houshold Couple']=0
    df.loc[couples,'Houshold Couple']=1


    df['Education Level'] = df['Education'].apply(education_level)


    return df

def data_cleaning(df):

    idx_null=df[df['Income'].isnull()]['Income'].index
    imp_mean = IterativeImputer(random_state=5)
    impute_income = imp_mean.fit_transform(df[['Income','Total Spend','Catalog Purchases','Store Purchases','NumWebVisitsMonth','Education Level']])
    df.iloc[idx_null,4] = pd.Series(impute_income[idx_null,0])
    
    # clean max age 
    mask_age = df['Age']>80
    df.mask(mask_age,inplace=True)

    # clean YOLO
    mask_yolo = df['Marital_Status']=='YOLO'
    df.mask(mask_yolo,inplace=True)

    # clean Absurd
    mask_absurd = df['Marital_Status'] == 'Absurd'
    df.mask(mask_absurd,inplace=True)

    # clean 0 Total purchase
    mask_purchase = df['Total Purchase'] == 0
    df.mask(mask_purchase,inplace=True)

    # clean income with 6666666
    mask_high_income = df['Income']>500000
    df.mask(mask_high_income,inplace=True)

    # clean ME
    mask_country = df['Country']=='ME'
    df.mask(mask_country,inplace=True)

    
    return df

def education_level(education):
    
    if education in ['PhD', 'Master']:
        return 3.0
    
    elif education in ['Graduation']:
        return 2.0
    
    elif education in ['2n Cycle Education']:
        return 1.0
    
    else:
        return 0.0
    