import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  Banknote,
  BarChart3,
  Building2,
  CircleDollarSign,
  Gauge,
  TrendingUp,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

const currencyFormatter = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
})

const numberFormatter = new Intl.NumberFormat('en-IN', {
  maximumFractionDigits: 0,
})

function formatCurrency(value) {
  return currencyFormatter.format(Number(value || 0))
}

function formatNumber(value) {
  return numberFormatter.format(Number(value || 0))
}

function sortMonthLabels(values) {
  return [...values].sort((left, right) => {
    const leftDate = new Date(`${left} 01T00:00:00`)
    const rightDate = new Date(`${right} 01T00:00:00`)
    return leftDate - rightDate
  })
}

function App() {
  const [activeTab, setActiveTab] = useState('home')
  const [dashboard, setDashboard] = useState({ summary: {}, historical_trend: [] })
  const [dataAudit, setDataAudit] = useState({
    dataset_kind: 'monthly_bank_level',
    available_columns: [],
    missing_data: {},
    date_range: {},
    records_loaded: 0,
    unique_banks: 0,
    source_files: 0,
    message: '',
  })
  const [analytics, setAnalytics] = useState({ top_banks: [], monthly_totals: [] })
  const [performance, setPerformance] = useState({ metrics: {}, comparison: [] })
  const [featureOptions, setFeatureOptions] = useState({ banks: [], months: [] })
  const [trendAnalysis, setTrendAnalysis] = useState({
    selected_bank: 'All Banks',
    available_banks: [],
    available_months: [],
    summary: {},
    series: [],
  })
  const [trendBank, setTrendBank] = useState('All Banks')
  const [trendStartMonth, setTrendStartMonth] = useState('')
  const [trendEndMonth, setTrendEndMonth] = useState('')
  const [history, setHistory] = useState([])
  const [dailyCapability, setDailyCapability] = useState({
    supports_daily_atm_forecast: false,
    message: 'Loading daily forecast capability…',
    missing_data: {},
    available_columns: [],
    atm_ids: [],
  })
  const [dailyForm, setDailyForm] = useState({
    atm_id: '',
    prediction_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10),
    available_cash_balance: 0,
  })
  const [dailyPrediction, setDailyPrediction] = useState(null)
  const [dailyError, setDailyError] = useState('')
  const [isFetchingDashboard, setIsFetchingDashboard] = useState(true)
  const [error, setError] = useState('')

  const fetchData = async () => {
    try {
      setError('')
      setIsFetchingDashboard(true)

      const [dashboardRes, analyticsRes, performanceRes, optionsRes, historyRes, dataAuditRes] = await Promise.all([
        fetch(`${API_URL}/api/dashboard`),
        fetch(`${API_URL}/api/analytics`),
        fetch(`${API_URL}/api/performance`),
        fetch(`${API_URL}/api/feature-options`),
        fetch(`${API_URL}/api/history`),
        fetch(`${API_URL}/api/data-audit`),
      ])

      if (!dashboardRes.ok || !analyticsRes.ok || !performanceRes.ok || !optionsRes.ok || !historyRes.ok || !dataAuditRes.ok) {
        const details = await Promise.all([
          dashboardRes.text(),
          analyticsRes.text(),
          performanceRes.text(),
          optionsRes.text(),
          historyRes.text(),
          dataAuditRes.text(),
        ])
        throw new Error(`Backend API error. Details: ${details.filter(Boolean).join(' | ') || 'unknown error'}`)
      }

      const dashboardData = await dashboardRes.json()
      const analyticsData = await analyticsRes.json()
      const performanceData = await performanceRes.json()
      const optionsData = await optionsRes.json()
      const historyData = await historyRes.json()
      const dataAuditData = await dataAuditRes.json()

      if (!dashboardData?.summary || Number(dashboardData.summary.total_records || 0) === 0) {
        throw new Error('The backend returned no real rows from the Excel dataset. Check the dataset files and backend logs.')
      }

      setDashboard(dashboardData)
      setAnalytics(analyticsData)
      setPerformance(performanceData)
      setFeatureOptions(optionsData)
      setHistory(historyData.records || [])
      setDataAudit(dataAuditData)

      const sortedMonths = sortMonthLabels(optionsData.months || [])
      const defaultStart = sortedMonths[0] || ''
      const defaultEnd = sortedMonths[sortedMonths.length - 1] || ''
      setTrendStartMonth((current) => current || defaultStart)
      setTrendEndMonth((current) => current || defaultEnd)
      setTrendBank((current) => current || 'All Banks')
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : 'Failed to fetch dashboard data.'
      setError(`Unable to load live data from the backend at ${API_URL}. ${message}`)
    } finally {
      setIsFetchingDashboard(false)
    }
  }

  const fetchDailyCapability = async () => {
    try {
      const response = await fetch(`${API_URL}/api/daily-capability`)
      if (!response.ok) {
        throw new Error('Unable to load daily forecast capability data.')
      }
      const payload = await response.json()
      setDailyCapability(payload)
      setDailyForm((current) => ({
        ...current,
        atm_id: current.atm_id || payload.atm_ids?.[0] || 'ATM-UNAVAILABLE',
        prediction_date: current.prediction_date || payload.date_range?.start || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10),
      }))
    } catch (loadError) {
      setDailyCapability({
        supports_daily_atm_forecast: false,
        message: loadError instanceof Error ? loadError.message : 'Daily ATM capability unavailable.',
        missing_data: {
          atm_identifier: true,
          daily_atm_history: true,
          holiday_calendar: true,
        },
        available_columns: [],
        atm_ids: [],
      })
    }
  }

  const fetchTrendAnalysis = async (bank = trendBank, start = trendStartMonth, end = trendEndMonth) => {
    try {
      const params = new URLSearchParams()
      if (bank && bank !== 'All Banks') params.set('bank_name', bank)
      if (start) params.set('start_month', start)
      if (end) params.set('end_month', end)
      const response = await fetch(`${API_URL}/api/trend-analysis${params.toString() ? `?${params.toString()}` : ''}`)
      if (!response.ok) {
        throw new Error('Trend analysis endpoint failed.')
      }
      const payload = await response.json()
      setTrendAnalysis(payload)
      if (payload.selected_bank) setTrendBank(payload.selected_bank)
    } catch (fetchError) {
      setTrendAnalysis({
        selected_bank: bank || 'All Banks',
        available_banks: featureOptions.banks || [],
        available_months: featureOptions.months || [],
        summary: {},
        series: [],
      })
    }
  }

  useEffect(() => {
    fetchData()
    fetchDailyCapability()
  }, [])

  useEffect(() => {
    if (featureOptions.months.length || featureOptions.banks.length || trendBank) {
      fetchTrendAnalysis()
    }
  }, [trendBank, trendStartMonth, trendEndMonth, featureOptions.months.length, featureOptions.banks.length])

  const pieData = useMemo(() => {
    const topBanks = analytics.top_banks || []
    return topBanks.length > 0 ? topBanks.slice(0, 5) : []
  }, [analytics])

  const handleDailyFieldChange = (event) => {
    const { name, value } = event.target
    setDailyForm((current) => ({
      ...current,
      [name]: name === 'available_cash_balance' ? Number(value || 0) : value,
    }))
  }

  const submitDailyForecast = async (event) => {
    event.preventDefault()
    setDailyError('')
    setDailyPrediction(null)

    try {
      const response = await fetch(`${API_URL}/api/predict-daily`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          atm_id: dailyForm.atm_id,
          prediction_date: dailyForm.prediction_date,
          available_cash_balance: dailyForm.available_cash_balance,
        }),
      })

      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail || 'Daily ATM forecast is unavailable for the current dataset.')
      }

      setDailyPrediction(payload.prediction || payload)
    } catch (submitError) {
      setDailyError(submitError instanceof Error ? submitError.message : 'Daily ATM forecast request failed.')
    }
  }

  const trendSeries = (trendAnalysis.series || []).map((item) => ({
    month: item.month_label || item.month || 'N/A',
    month_label: item.month_label || item.month || 'N/A',
    cash_requirement_inr: Number(item.cash_requirement_inr) || 0,
    cash_withdrawal_total_atm_volume: Number(item.cash_withdrawal_total_atm_volume) || 0,
  }))
  const trendSummary = trendAnalysis.summary || {}
  const availableDailyAtms = dailyCapability.atm_ids || []

  const summaryCards = [
    {
      label: 'Total bank records',
      value: formatNumber(dashboard.summary?.total_records || 0),
      icon: Building2,
      accent: 'blue',
    },
    {
      label: 'Unique banks',
      value: formatNumber(dashboard.summary?.unique_banks || 0),
      icon: Banknote,
      accent: 'violet',
    },
    {
      label: 'Avg. cash requirement',
      value: formatCurrency(dashboard.summary?.average_cash_requirement_inr || 0),
      icon: CircleDollarSign,
      accent: 'green',
    },
    {
      label: 'Peak month',
      value: dashboard.summary?.peak_month || 'N/A',
      icon: TrendingUp,
      accent: 'orange',
    },
  ]

  const navTabs = [
    { key: 'home', label: 'Home' },
    { key: 'data-explorer', label: 'Data Explorer' },
    { key: 'prediction', label: 'Prediction' },
  ]

  const renderHomeView = () => (
    <>
      <section className="overview-panel panel">
        <div>
          <p className="eyebrow">Overview</p>
          <h2>ATM cash requirement planning dashboard</h2>
        </div>
        <div className="overview-copy">
          <p>
            Monitor monthly RBI cash demand patterns across banks, review the strongest demand months, and understand where the
            model is performing best before producing a forecast.
          </p>
          <p>
            The platform keeps the real monthly bank dataset separate from the synthetic daily ATM demo used for future-date testing.
          </p>
        </div>
      </section>

      <section className="summary-grid">
        {summaryCards.map(({ label, value, icon: Icon, accent }) => (
          <article key={label} className={`summary-card ${accent}`}>
            <div className="summary-icon">
              <Icon size={18} />
            </div>
            <div>
              <p>{label}</p>
              <strong>{value}</strong>
            </div>
          </article>
        ))}
      </section>

      <section className="panel-grid two-up">
        <article className="panel trend-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Trend analysis</p>
              <h2>Monthly cash withdrawal trend</h2>
            </div>
            <BarChart3 size={18} />
          </div>

          <div className="trend-controls">
            <label className="compact-label">
              <span>Bank</span>
              <select value={trendBank} onChange={(event) => setTrendBank(event.target.value)}>
                <option value="All Banks">All Banks</option>
                {(trendAnalysis.available_banks.length ? trendAnalysis.available_banks : featureOptions.banks).map((bank) => (
                  <option key={bank} value={bank}>{bank}</option>
                ))}
              </select>
            </label>

            <label className="compact-label">
              <span>Start month</span>
              <select value={trendStartMonth} onChange={(event) => setTrendStartMonth(event.target.value)}>
                {(trendAnalysis.available_months.length ? trendAnalysis.available_months : featureOptions.months).map((month) => (
                  <option key={month} value={month}>{month}</option>
                ))}
              </select>
            </label>

            <label className="compact-label">
              <span>End month</span>
              <select value={trendEndMonth} onChange={(event) => setTrendEndMonth(event.target.value)}>
                {(trendAnalysis.available_months.length ? trendAnalysis.available_months : featureOptions.months).map((month) => (
                  <option key={month} value={month}>{month}</option>
                ))}
              </select>
            </label>
          </div>

          {trendSeries.length > 0 ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={trendSeries}>
                  <CartesianGrid stroke="#e5e7eb" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 10 }} minTickGap={12} />
                  <YAxis tickFormatter={(value) => `${(Number(value || 0) / 1e9).toFixed(1)}B`} width={72} />
                  <Tooltip
                    formatter={(value) => formatCurrency(Number(value || 0))}
                    labelFormatter={(label) => `Month: ${label}`}
                  />
                  <Line type="monotone" dataKey="cash_requirement_inr" stroke="#7367f0" strokeWidth={3} dot={{ r: 3 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="chart-empty">
              <strong>No trend data available</strong>
              <p>There are no matching records for the selected bank and month range.</p>
            </div>
          )}

          <div className="trend-summary">
            <div className="trend-summary-card">
              <span>Total ATM cash withdrawal value</span>
              <strong>{formatCurrency(trendSummary.total_cash_requirement_inr || 0)}</strong>
            </div>
            <div className="trend-summary-card">
              <span>Total ATM cash withdrawal volume</span>
              <strong>{formatNumber(trendSummary.total_cash_withdrawal_volume || 0)}</strong>
            </div>
            <div className="trend-summary-card">
              <span>Highest month</span>
              <strong>{trendSummary.peak_month || 'N/A'}</strong>
            </div>
            <div className="trend-summary-card">
              <span>Change vs previous month</span>
              <strong>
                {trendSummary.percentage_change_vs_previous_month === null || trendSummary.percentage_change_vs_previous_month === undefined
                  ? 'N/A'
                  : `${Number(trendSummary.percentage_change_vs_previous_month).toFixed(1)}%`}
              </strong>
            </div>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Market outlook</p>
              <h2>Top banks</h2>
            </div>
            <Activity size={18} />
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={analytics.top_banks || []}>
                <CartesianGrid stroke="#e5e7eb" vertical={false} />
                <XAxis dataKey="bank_name" tick={{ fontSize: 10 }} />
                <YAxis tickFormatter={(value) => `${(value / 1e9).toFixed(1)}B`} />
                <Tooltip formatter={(value) => formatCurrency(value)} />
                <Bar dataKey="average_cash_requirement_inr" radius={[8, 8, 0, 0]} fill="#14b8a6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="panel-grid three-up">
        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Model health</p>
              <h2>Performance metrics</h2>
            </div>
            <Gauge size={18} />
          </div>
          <div className="metric-stack">
            <div className="metric-row">
              <span>Model</span>
              <strong>{performance.model_name || 'RandomForestRegressor'}</strong>
            </div>
            <div className="metric-row">
              <span>MAE</span>
              <strong>{formatCurrency(performance.metrics?.mae || 0)}</strong>
            </div>
            <div className="metric-row">
              <span>RMSE</span>
              <strong>{formatCurrency(performance.metrics?.rmse || 0)}</strong>
            </div>
            <div className="metric-row">
              <span>R² score</span>
              <strong>{Number(performance.metrics?.r2_score || 0).toFixed(4)}</strong>
            </div>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Share of demand</p>
              <h2>Bank mix</h2>
            </div>
            <Activity size={18} />
          </div>
          <div className="donut-wrap">
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={pieData} dataKey="average_cash_requirement_inr" nameKey="bank_name" innerRadius={42} outerRadius={70} paddingAngle={3}>
                  {pieData.map((entry, index) => (
                    <Cell key={`${entry.bank_name}-${index}`} fill={['#7367f0', '#14b8a6', '#f59e0b', '#ec4899', '#0ea5e9'][index % 5]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => formatCurrency(value)} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>
    </>
  )

  const renderDataExplorerView = () => (
    <>
      <section className="panel-grid two-up">
        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Data explorer</p>
              <h2>Dataset overview</h2>
            </div>
            <Building2 size={18} />
          </div>
          <div className="audit-grid">
            <div className="audit-metric">
              <span>Records loaded</span>
              <strong>{formatNumber(dataAudit.records_loaded || dashboard.summary?.total_records || 0)}</strong>
            </div>
            <div className="audit-metric">
              <span>Unique banks</span>
              <strong>{formatNumber(dataAudit.unique_banks || dashboard.summary?.unique_banks || 0)}</strong>
            </div>
            <div className="audit-metric">
              <span>Source files</span>
              <strong>{formatNumber(dataAudit.source_files || 0)}</strong>
            </div>
            <div className="audit-metric">
              <span>Data kind</span>
              <strong>{dataAudit.dataset_kind || 'monthly_bank_level'}</strong>
            </div>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Audit</p>
              <h2>Data quality checks</h2>
            </div>
            <Gauge size={18} />
          </div>
          <div className="audit-list">
            {Object.entries(dataAudit.missing_data || {}).map(([key, value]) => (
              <div key={key} className="audit-row">
                <span>{key.replace(/_/g, ' ')}</span>
                <strong className={value ? 'warn' : 'ok'}>{value ? 'Missing' : 'Present'}</strong>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="panel-grid two-up">
        <article className="panel trend-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Trend analysis</p>
              <h2>Monthly cash withdrawal trend</h2>
            </div>
            <BarChart3 size={18} />
          </div>
          {trendSeries.length > 0 ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={trendSeries}>
                  <CartesianGrid stroke="#e5e7eb" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 10 }} minTickGap={12} />
                  <YAxis tickFormatter={(value) => `${(Number(value || 0) / 1e9).toFixed(1)}B`} width={72} />
                  <Tooltip formatter={(value) => formatCurrency(Number(value || 0))} labelFormatter={(label) => `Month: ${label}`} />
                  <Line type="monotone" dataKey="cash_requirement_inr" stroke="#7367f0" strokeWidth={3} dot={{ r: 3 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="chart-empty">
              <strong>No trend data available</strong>
              <p>There are no matching records for the selected bank and month range.</p>
            </div>
          )}
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Bank mix</p>
              <h2>Top banks</h2>
            </div>
            <Activity size={18} />
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={analytics.top_banks || []}>
                <CartesianGrid stroke="#e5e7eb" vertical={false} />
                <XAxis dataKey="bank_name" tick={{ fontSize: 10 }} />
                <YAxis tickFormatter={(value) => `${(value / 1e9).toFixed(1)}B`} />
                <Tooltip formatter={(value) => formatCurrency(value)} />
                <Bar dataKey="average_cash_requirement_inr" radius={[8, 8, 0, 0]} fill="#14b8a6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Columns</p>
            <h2>Available dataset fields</h2>
          </div>
        </div>
        <div className="pill-list">
          {(dataAudit.available_columns || []).map((column) => (
            <span key={column} className="pill">{column}</span>
          ))}
        </div>
      </section>
    </>
  )

  const renderPredictionView = () => (
    <>
      <section className="panel daily-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Daily cash forecast</p>
            <h2>Individual ATM forecast</h2>
          </div>
        </div>

        

        <form onSubmit={submitDailyForecast} className="prediction-form">
          <div className="daily-grid">
            <label>
              <span>ATM</span>
              <select
                name="atm_id"
                value={dailyForm.atm_id}
                onChange={handleDailyFieldChange}
                disabled={!dailyCapability.supports_daily_atm_forecast}
              >
                {availableDailyAtms.length > 0 ? (
                  availableDailyAtms.map((atmId) => (
                    <option key={atmId} value={atmId}>{atmId}</option>
                  ))
                ) : (
                  <option value="ATM-UNAVAILABLE">ATM identifiers unavailable in this dataset</option>
                )}
              </select>
            </label>

            <label>
              <span>Prediction date</span>
              <input
                type="date"
                name="prediction_date"
                value={dailyForm.prediction_date}
                onChange={handleDailyFieldChange}
                disabled={!dailyCapability.supports_daily_atm_forecast}
                min={dailyCapability.date_range?.start || ''}
                max={dailyCapability.date_range?.end || ''}
              />
            </label>

            
          </div>

          <button type="submit" className="submit-button" disabled={!dailyCapability.supports_daily_atm_forecast}>
            {dailyCapability.supports_daily_atm_forecast ? 'Predict synthetic daily ATM cash' : 'Daily forecast unavailable'}
          </button>
        </form>

        {dailyError ? <div className="alert-box daily-error">{dailyError}</div> : null}

        {dailyPrediction ? (
          <div className="daily-result">
            <div className="metric-row"><span>ATM</span><strong>{dailyPrediction.atm_id}</strong></div>
            <div className="metric-row"><span>Predicted cash</span><strong>{formatCurrency(dailyPrediction.predicted_cash_requirement_inr)}</strong></div>
            <div className="metric-row"><span>Dataset</span><strong>{dailyPrediction.dataset_kind || 'synthetic_daily_demo'}</strong></div>
            <div className="metric-row"><span>Note</span><strong>{dailyPrediction.message || 'Synthetic demo forecast only'}</strong></div>
          </div>
        ) : null}

        
      </section>

      <section className="dashboard-bottom">
        <article className="panel table-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">History</p>
              <h2>Prediction records</h2>
            </div>
          </div>

          <div className="history-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Bank</th>
                  <th>Month</th>
                  <th>Cash need</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 6).map((row, idx) => (
                  <tr key={`${row.bank_name}-${row.month}-${idx}`}>
                    <td>{row.bank_name}</td>
                    <td>{row.month}</td>
                    <td>{formatCurrency(row.cash_requirement_inr)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </section>
    </>
  )

  return (
    <div className="page-shell">
      <header className="topbar">
        <div className="brand-wrap">
          <div className="brand-mark">
            <Banknote size={18} />
          </div>
          <div className="brand-copy">
            <span className="brand-tag">ATM</span>
            <h1>Cash Prediction</h1>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Main navigation">
          {navTabs.map((tab) => (
            <button
              key={tab.key}
              type="button"
              className={`nav-button ${activeTab === tab.key ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      {isFetchingDashboard ? <div className="alert-box">Loading live dashboard data…</div> : null}
      {error ? <div className="alert-box">{error}</div> : null}

      {!isFetchingDashboard && !error && Number(dashboard.summary?.total_records || 0) === 0 ? (
        <div className="alert-box">No real records were loaded from the Excel dataset. The backend returned an empty dataset.</div>
      ) : null}

      {!isFetchingDashboard && !error && Number(dashboard.summary?.total_records || 0) > 0 ? (
        <>
          {activeTab === 'home' && renderHomeView()}
          {activeTab === 'data-explorer' && renderDataExplorerView()}
          {activeTab === 'prediction' && renderPredictionView()}
        </>
      ) : null}
    </div>
  )
}

export default App
