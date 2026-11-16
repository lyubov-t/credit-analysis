import pandas as pd
import matplotlib.pyplot as plt

# ---------- Загрузка ----------
apps  = pd.read_csv('applications.csv').drop_duplicates('application_id')
loans = pd.read_csv('loans.csv').drop_duplicates('loan_id')
pay   = pd.read_csv('payment_status.csv')

# ---------- Очистка ----------
apps['product'] = apps['product'].str.strip().str.lower()
apps['region']  = apps['region'].str.strip().str.replace('г. ', '', regex=False)
apps['app_month'] = pd.to_datetime(apps['application_dt']).dt.to_period('M').astype(str)
apps['requested_amount'] = pd.to_numeric(apps['requested_amount'], errors='coerce')
loans['issued_amount']   = pd.to_numeric(loans['issued_amount'], errors='coerce')

# ---------- Последний снимок на кредит ----------
pay['snapshot_dt'] = pd.to_datetime(pay['snapshot_dt'], errors='coerce')
last = (pay.sort_values('snapshot_dt')
           .groupby('loan_id').tail(1)
           .groupby('loan_id').agg(max_dpd=('days_past_due', 'max'))
           .reset_index())

# ---------- Витрина на уровне заявки ----------
df = (apps.merge(loans[['loan_id','application_id','issued_amount','term_months']],
                 on='application_id', how='left')
         .merge(last, on='loan_id', how='left'))

df['has_loan']   = df['loan_id'].notna()
df['approved']   = df['decision'].eq('approved')
df['dpd30']      = df['max_dpd'].ge(30)
df['dpd30_flag'] = (df['has_loan'] & df['dpd30']).astype(int)
df['req_pos']    = df['requested_amount'].where(df['requested_amount'] > 0)
df['iss_pos']    = df['issued_amount'].where(df['issued_amount'] > 0)

# ---------- Метрики ----------
g = df.groupby(['app_month','product'])
rep = g.agg(
    applications  = ('application_id', 'size'),
    approved      = ('approved',       'sum'),
    issued        = ('has_loan',       'sum'),
    avg_requested = ('req_pos',        'mean'),
    avg_issued    = ('iss_pos',        'mean'),
    dpd30_loans   = ('dpd30_flag',     'sum'),
).reset_index()

rep['approve_rate']  = (rep['approved'] / rep['applications']).round(3)
rep['issue_rate']    = (rep['issued']   / rep['approved'].replace(0, pd.NA)).round(3)
rep['dpd30_share']   = (rep['dpd30_loans'] / rep['issued'].replace(0, pd.NA)).round(3)
rep['avg_requested'] = rep['avg_requested'].round(0)
rep['avg_issued']    = rep['avg_issued'].round(0)

print(rep.to_string(index=False))
rep.to_csv('metrics_by_month_product.csv', index=False)

# ---------- График ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, c, t in [(ax[0], 'approve_rate', 'Доля одобренных'),
                (ax[1], 'dpd30_share',  'DPD 30+ среди выданных')]:
    rep.pivot(index='app_month', columns='product', values=c) \
       .plot(kind='bar', ax=a, title=t, ylim=(0,1), rot=45)
    a.set_ylabel('доля'); a.grid(axis='y', alpha=.3)
plt.tight_layout()
plt.savefig('charts.png', dpi=110)

print('\n✓ Готово: metrics_by_month_product.csv, charts.png')