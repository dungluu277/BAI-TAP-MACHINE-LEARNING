"""Hàm dùng chung cho các notebook EDA (Ames Housing / Kaggle House Prices)."""
import os, re, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA, FIG, RES = (os.path.join(ROOT, d) for d in ('data', 'figures', 'results'))
for d in (FIG, RES): os.makedirs(d, exist_ok=True)
sns.set_theme(style='whitegrid', font_scale=0.95)
plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 140, 'savefig.bbox': 'tight'})

# Phân loại 79 biến theo paper De Cock (23 nominal, 23 ordinal, 14 discrete, 19 continuous + SalePrice)
NOMINAL = ['MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood','Condition1','Condition2',
  'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating',
  'CentralAir','GarageType','MiscFeature','SaleType','SaleCondition']
ORDINAL = ['LotShape','Utilities','LandSlope','OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual','BsmtCond',
  'BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','Electrical','KitchenQual','Functional','FireplaceQu',
  'GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence']
DISCRETE = ['YearBuilt','YearRemodAdd','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
  'TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','MoSold','YrSold']
CONTINUOUS = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
  'LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']
GROUP_OF = {**{c:'nominal' for c in NOMINAL}, **{c:'ordinal' for c in ORDINAL},
            **{c:'discrete' for c in DISCRETE}, **{c:'continuous' for c in CONTINUOUS}}
# Cột mà NA trong data_description.txt nghĩa là "không có" (No alley, No basement, No garage...)
NA_MEANS_NONE = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType',
  'GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']

def load():
    """Đọc dữ liệu đúng cách: chỉ chuỗi 'NA' là thiếu; chữ 'None' (vd MasVnrType) là một nhóm hợp lệ."""
    kw = dict(keep_default_na=False, na_values=['NA'])
    return pd.read_csv(f'{DATA}/train.csv', **kw), pd.read_csv(f'{DATA}/test.csv', **kw)

def descriptions():
    d = {}
    for line in open(f'{DATA}/data_description.txt', encoding='latin-1'):
        m = re.match(r'^(\w+):\s*(.*)$', line.strip())
        if m: d[m.group(1)] = m.group(2).strip()
    return d

def save_fig(name):
    plt.tight_layout(); plt.savefig(f'{FIG}/{name}.png'); plt.show()

def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
def save_json(name, obj):
    json.dump(_clean(obj), open(f'{RES}/{name}.json', 'w'), ensure_ascii=False, indent=1)
