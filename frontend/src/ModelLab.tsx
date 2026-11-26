import { useEffect, useState } from 'react'
import { api } from './api'
import type { Movie, Pick } from './main'

type Model = 'itemknn' | 'neural'
type Rating = { movie: Movie; stars: number }
type Mode = 'model' | 'rating'
type Comparison = { before: Pick[]; after: Pick[] }
type Models = { active: Model; available: Model[] }
type Result = { users: number; ndcg_at_10: Record<Model, number>;
  paired_differences: { neural_minus_itemknn: { difference: number; ci95: number[] } } }
type Report = { selection: Model; code_sha: string; metrics: { validation: Result; test: Result } }

const labels: Record<Model, string> = { itemknn: 'ItemKNN', neural: 'Neural' }

function Changes({ comparison }: { comparison: Comparison }) {
  const before = new Map(comparison.before.map((pick, i) => [pick.movie.id, i + 1]))
  const after = new Map(comparison.after.map((pick, i) => [pick.movie.id, i + 1]))
  const movies = [...comparison.after, ...comparison.before.filter(pick => !after.has(pick.movie.id))]
  return <div className="lab-scroll"><table className="lab-table">
    <caption>Top-ten changes</caption><thead><tr><th>Film</th><th>Before</th><th>After</th><th>Change</th></tr></thead>
    <tbody>{movies.map(({ movie }) => {
      const first = before.get(movie.id)
      const second = after.get(movie.id)
      const change = first === undefined ? 'Entered' : second === undefined ? 'Left' : first === second ? 'Same' : first > second ? `Up ${first - second}` : `Down ${second - first}`
      return <tr key={movie.id}><th scope="row">{movie.title}</th><td>{first ?? '—'}</td><td>{second ?? '—'}</td><td>{change}</td></tr>
    })}</tbody>
  </table></div>
}

export default function ModelLab({ ratings }: { ratings: Rating[] }) {
  const [profile, setProfile] = useState(ratings)
  const [settings, setSettings] = useState<Models>()
  const [report, setReport] = useState<Report>()
  const [mode, setMode] = useState<Mode>('model')
  const [model, setModel] = useState<Model>('itemknn')
  const [movieId, setMovieId] = useState<number>()
  const [stars, setStars] = useState(2)
  const [comparison, setComparison] = useState<Comparison>()
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    api<Models>('/models', { signal: controller.signal }).then(setSettings).catch(reason => {
      if (!controller.signal.aborted) setError(reason.message)
    })
    api<Report>('/evidence/release-3', { signal: controller.signal }).then(setReport).catch(() => undefined)
    return () => controller.abort()
  }, [])

  async function example() {
    setPending(true); setError(''); setComparison(undefined)
    try {
      const movies = await api<Movie[]>('/prompts')
      setProfile(movies.slice(0, 3).map((movie, i) => ({ movie, stars: 5 - i })))
      setMovieId(undefined)
    } catch (reason) { setError((reason as Error).message) }
    finally { setPending(false) }
  }

  async function compare() {
    setPending(true); setError(''); setComparison(undefined)
    try {
      const target = movieId ?? profile[0]?.movie.id
      setComparison(await api<Comparison>('/compare', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
          ratings: profile.map(({ movie, stars }) => ({ movie_id: movie.id, stars })),
          before_model: mode === 'model' ? 'itemknn' : model,
          after_model: mode === 'model' ? 'neural' : model,
          edit: mode === 'rating' ? { movie_id: target, stars } : null,
        }),
      }))
    } catch (reason) { setError((reason as Error).message) }
    finally { setPending(false) }
  }

  return <section className="benchmark lab" aria-labelledby="lab-title">
    <h1 id="lab-title">Model Lab</h1>
    <p className="lead">Change one rating or compare two models. See which films move in the top ten.</p>
    {settings && <p>Live feed model <strong>{labels[settings.active]}</strong></p>}
    <div className="panel lab-controls">
      <div className="panel-top"><h2>Taste profile</h2><button className="text-button" type="button" onClick={example} disabled={pending}>Use example ratings</button></div>
      {profile.length ? <ul className="lab-ratings">{profile.map(({ movie, stars }) => <li key={movie.id}>{movie.title} <strong>{stars} ★</strong></li>)}</ul>
        : <p>Rate movies in Discover or load an example profile.</p>}
      <div className="lab-fields">
        <label htmlFor="compare-mode">Compare<select id="compare-mode" value={mode} disabled={pending} onChange={event => { setMode(event.target.value as Mode); setComparison(undefined) }}>
          <option value="model">Switch models</option><option value="rating">Edit one rating</option>
        </select></label>
        {mode === 'rating' && <>
          <label htmlFor="edit-model">Model<select id="edit-model" value={model} disabled={pending} onChange={event => { setModel(event.target.value as Model); setComparison(undefined) }}>
            {(settings?.available ?? ['itemknn']).map(choice => <option key={choice} value={choice}>{labels[choice as Model]}</option>)}
          </select></label>
          <label htmlFor="edit-film">Film<select id="edit-film" value={movieId ?? profile[0]?.movie.id ?? ''} disabled={pending} onChange={event => { setMovieId(Number(event.target.value)); setComparison(undefined) }}>
            {profile.map(({ movie }) => <option key={movie.id} value={movie.id}>{movie.title}</option>)}
          </select></label>
          <label htmlFor="edit-stars">New rating<select id="edit-stars" value={stars} disabled={pending} onChange={event => { setStars(Number(event.target.value)); setComparison(undefined) }}>
            {[1, 2, 3, 4, 5].map(value => <option key={value} value={value}>{value} ★</option>)}
          </select></label>
        </>}
        <button type="button" className="lab-run" onClick={compare} disabled={pending || !profile.length || !settings || (mode === 'model' && !settings.available.includes('neural'))}>Compare top ten</button>
      </div>
      {pending && <p role="status">Loading comparison…</p>}
      {error && <p role="alert">{error}</p>}
    </div>
    {comparison && <Changes comparison={comparison} />}
    {report && <section className="lab-results"><h2>Held-out model results</h2>
      <p>Both models ranked the same candidate movies with the same content features and fixed prompts.</p>
      <div className="lab-scroll"><table className="lab-table">
        <thead><tr><th>Model</th><th>Validation NDCG@10</th><th>Test NDCG@10</th></tr></thead>
        <tbody>{(['itemknn', 'neural'] as const).map(choice => <tr key={choice}><th scope="row">{labels[choice]}</th><td>{report.metrics.validation.ndcg_at_10[choice].toFixed(4)}</td><td>{report.metrics.test.ndcg_at_10[choice].toFixed(4)}</td></tr>)}</tbody>
      </table></div>
      <p className="movie-meta">Test viewers {report.metrics.test.users.toLocaleString()} · Neural minus ItemKNN {report.metrics.test.paired_differences.neural_minus_itemknn.difference.toFixed(4)} · 95% CI [{report.metrics.test.paired_differences.neural_minus_itemknn.ci95.map(value => value.toFixed(4)).join(', ')}]</p>
      <p className="movie-meta">Source revision <code>{report.code_sha}</code></p>
    </section>}
  </section>
}
