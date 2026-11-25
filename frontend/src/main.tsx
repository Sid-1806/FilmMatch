import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

type Movie = { id: number; title: string; year: number | null; genres: string[]; rating_count: number }
type Pick = { movie: Movie; because: string | null; signal: 'ratings' | 'content' | 'popular' }
type Profile = { ratings: Record<number, { movie: Movie; stars: number }>; answered: number[] }
type Metrics = {
  assigned_users: number; eligible_users: number; popularity_ndcg_at_10?: number;
  itemknn_ndcg_at_10?: number; paired_lift?: number; paired_bootstrap_95_ci?: number[];
  simulated_skip_rate?: number;
}
type Evidence = { dataset: string; sampled_users: number; code_sha: string; seed: number; metrics: Record<string, Metrics> }
type Comparison = { users: number; ndcg_at_10: Record<string, number>; skip_rate?: Record<string, number>;
  paired_differences: Record<string, { difference: number; ci95: number[] }> }
type ContextEvidence = { code_sha: string; prompt_replay: Record<string, Comparison>;
  cold_items: Record<string, Comparison>; descriptions: number; catalog_size: number }

const empty: Profile = { ratings: {}, answered: [] }
const storageKey = 'filmmatch.profile.v2'
const promptLimit = 5

function initialProfile(): Profile {
  try {
    const data = JSON.parse(localStorage.getItem(storageKey) ?? 'null') as Profile | null
    if (data?.ratings && Array.isArray(data.answered)) return data
    const previous = JSON.parse(localStorage.getItem('filmmatch.profile.v1') ?? 'null') as Profile | null
    return previous?.ratings ? { ratings: previous.ratings, answered: [] } : empty
  } catch { return empty }
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const result = await fetch(`/api${path}`, options)
  if (!result.ok) throw new Error(result.status === 404 ? 'This report is pending publication.' : 'Could not load this request. Try again.')
  return result.json() as Promise<T>
}

function Stars({ movie, value, onRate }: { movie: Movie; value?: number; onRate: (value: number) => void }) {
  return <div className="stars" role="group" aria-label={`Rate ${movie.title}`}>
    {[1, 2, 3, 4, 5].map(stars => <button key={stars} type="button" aria-pressed={value === stars}
      aria-label={`${stars} ${stars === 1 ? 'star' : 'stars'}`} className={value && stars <= value ? 'filled' : ''}
      onClick={() => onRate(stars)}>★</button>)}
  </div>
}

function MovieMeta({ movie }: { movie: Movie }) {
  return <span className="movie-meta">{[movie.year, ...movie.genres.slice(0, 2)].filter(Boolean).join(' · ')}</span>
}

function Benchmark() {
  const [report, setReport] = useState<Evidence>()
  const [context, setContext] = useState<ContextEvidence>()
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    api<Evidence>('/evidence', { signal: controller.signal }).then(setReport).catch(reason => {
      if (!controller.signal.aborted) setError(reason.message)
    })
    api<ContextEvidence>('/evidence/release-2', { signal: controller.signal }).then(setContext).catch(() => undefined)
    return () => controller.abort()
  }, [])
  return <section className="benchmark" aria-labelledby="benchmark-title">
    <h1 id="benchmark-title">The benchmark</h1>
    <p className="lead">How well do popularity and ItemKNN rank films that held-out viewers rated highly later?</p>
    {error && <p role="status">{error}</p>}
    {!error && !report && <p role="status">Loading benchmark…</p>}
    {report && <>
      <p>{report.dataset} · {report.sampled_users.toLocaleString()} sampled users · Seed {report.seed}</p>
      <div className="benchmark-grid">{Object.entries(report.metrics).map(([name, run]) => <article key={name} className="panel metric">
        <h2>{name === 'test' ? 'Test' : 'Validation'}</h2>
        <p>{run.eligible_users.toLocaleString()} of {run.assigned_users.toLocaleString()} assigned users eligible</p>
        {run.itemknn_ndcg_at_10 !== undefined && <dl>
          <div><dt>Popularity NDCG@10</dt><dd>{run.popularity_ndcg_at_10?.toFixed(4)}</dd></div>
          <div><dt>ItemKNN NDCG@10</dt><dd>{run.itemknn_ndcg_at_10.toFixed(4)}</dd></div>
          <div><dt>Paired difference, 95% CI</dt><dd>{run.paired_lift?.toFixed(4)} [{run.paired_bootstrap_95_ci?.map(value => value.toFixed(4)).join(', ')}]</dd></div>
          <div><dt>Simulated skips</dt><dd>{((run.simulated_skip_rate ?? 0) * 100).toFixed(1)}%</dd></div>
        </dl>}
      </article>)}</div>
      <p className="movie-meta">Baseline source revision <code>{report.code_sha}</code></p>
    </>}
    {context && <>
      <div className="benchmark-grid">
        <ComparisonCard title="Five-prompt replay" results={context.prompt_replay.test} />
        <ComparisonCard title="Movies without ratings" results={context.cold_items.test} />
      </div>
      <p className="movie-meta">Plot descriptions matched for {context.descriptions.toLocaleString()} of {context.catalog_size.toLocaleString()} catalog titles. Source revision <code>{context.code_sha}</code></p>
    </>}
  </section>
}

function ComparisonCard({ title, results }: { title: string; results: Comparison }) {
  return <article className="panel metric">
    <h2>{title}</h2><p>{results.users.toLocaleString()} eligible test users</p>
    <dl>{Object.entries(results.ndcg_at_10).map(([key, value]) => <div key={key}>
      <dt>{key.replaceAll('_', ' ')} NDCG@10</dt><dd>{value.toFixed(4)}</dd></div>)}
      {Object.entries(results.skip_rate ?? {}).map(([key, value]) => <div key={key}>
        <dt>{key.replaceAll('_', ' ')} simulated skips</dt><dd>{(100 * value).toFixed(1)}%</dd></div>)}
      {Object.entries(results.paired_differences).map(([key, value]) => <div key={key}>
        <dt>{key.replaceAll('_', ' ')} (95% CI)</dt><dd>{value.difference.toFixed(4)} [{value.ci95.map(n => n.toFixed(4)).join(', ')}]</dd></div>)}
    </dl>
  </article>
}

function App() {
  const [tab, setTab] = useState<'discover' | 'benchmark'>('discover')
  const [profile, setProfile] = useState(initialProfile)
  const [prompt, setPrompt] = useState<Movie | null>()
  const [promptError, setPromptError] = useState('')
  const [genres, setGenres] = useState<string[]>([])
  const [picks, setPicks] = useState<Pick[]>([])
  const [search, setSearch] = useState('')
  const [results, setResults] = useState<Movie[]>([])
  const [selected, setSelected] = useState<Movie>()
  const [genre, setGenre] = useState('')
  const [decade, setDecade] = useState('')
  const [initialError, setInitialError] = useState('')
  const [searchError, setSearchError] = useState('')
  const [recommendError, setRecommendError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    api<{ genres: string[] }>('/catalog', { signal: controller.signal }).then(catalog => setGenres(catalog.genres))
      .catch(reason => { if (!controller.signal.aborted) setInitialError(reason.message) })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(profile)) } catch { /* Session still works in memory. */ }
  }, [profile])

  useEffect(() => {
    const controller = new AbortController()
    setPromptError('')
    if (profile.answered.length >= promptLimit) { setPrompt(null); return }
    setPrompt(undefined)
    api<Movie | null>('/prompt', { method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ shown: profile.answered, ratings: Object.values(profile.ratings).map(({ movie, stars }) => ({ movie_id: movie.id, stars })) }) })
      .then(setPrompt).catch(reason => { if (!controller.signal.aborted) setPromptError(reason.message) })
    return () => controller.abort()
  }, [profile])

  useEffect(() => {
    const controller = new AbortController()
    const query = search.trim()
    setResults([]); setSearchError('')
    if (!query) return
    const timeout = window.setTimeout(() => {
      api<Movie[]>(`/search?q=${encodeURIComponent(query)}`, { signal: controller.signal })
        .then(movies => { setResults(movies); if (!movies.length) setSearchError('No matching films.') })
        .catch(reason => { if (!controller.signal.aborted) setSearchError(reason.message) })
    }, 140)
    return () => { window.clearTimeout(timeout); controller.abort() }
  }, [search])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true); setRecommendError('')
    api<Pick[]>('/recommend', { method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ratings: Object.values(profile.ratings).map(({ movie, stars }) => ({ movie_id: movie.id, stars })),
        genre: genre || null, year_from: decade ? Number(decade) : null, year_to: decade ? Number(decade) + 9 : null }) })
      .then(setPicks).catch(reason => { if (!controller.signal.aborted) setRecommendError(reason.message) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [profile.ratings, genre, decade])

  const progress = profile.answered.length
  const ratingCount = Object.keys(profile.ratings).length
  const choose = (movie: Movie, stars: number) => {
    setProfile(current => ({ ratings: { ...current.ratings, [movie.id]: { movie, stars } },
      answered: movie.id === prompt?.id ? [...current.answered, movie.id] : current.answered }))
    if (movie.id === prompt?.id) setPrompt(undefined)
    setSelected(undefined)
  }
  const reset = () => { setProfile(empty); setSelected(undefined); setSearch(''); setGenre(''); setDecade('') }

  return <div className="site">
    <header className="header"><div className="shell topbar">
      <button type="button" className="brand" onClick={() => setTab('discover')}><span className="brand-mark" aria-hidden="true">▶</span> FilmMatch</button>
      <nav aria-label="Main navigation">
        <button type="button" aria-current={tab === 'discover' ? 'page' : undefined} onClick={() => setTab('discover')}>Discover</button>
        <button type="button" aria-current={tab === 'benchmark' ? 'page' : undefined} onClick={() => setTab('benchmark')}>Benchmark</button>
      </nav>
    </div></header>
    <main className="shell">
      {tab === 'benchmark' ? <Benchmark /> : <>
        <section className="hero">
          <div className="hero-copy"><h1>Find a film<br />worth <em>watching.</em></h1>
            <p className="lead">Rate a few movies you know. Get a shortlist that gets closer to your taste with every answer.</p></div>
          <div className="screen" aria-hidden="true"><div className="screen-letter">F<span>M</span></div></div>
        </section>
        {initialError && <p role="alert">{initialError}</p>}
        <section className="taste-grid" aria-label="Build your taste profile">
          <div className="panel taste">
            <div className="panel-top"><h2>Rate a movie</h2><span className="progress-label">{progress} / {promptLimit}</span></div>
            <div className="progress" aria-hidden="true">{Array.from({ length: promptLimit }, (_, i) => <span key={i} className={i < progress ? 'done' : ''} />)}</div>
            {promptError && <p className="prompt" role="alert">{promptError}</p>}
            {prompt === undefined && !promptError && <p className="prompt" role="status">Choosing a movie…</p>}
            {prompt && <div className="prompt"><h3>{prompt.title}</h3><MovieMeta movie={prompt} />
              <div className="prompt-actions"><Stars movie={prompt} onRate={stars => choose(prompt, stars)} />
                <button type="button" className="text-button" onClick={() => {
                  setPrompt(undefined)
                  setProfile(current => ({ ...current, answered: [...current.answered, prompt.id] }))
                }}>Haven’t seen it</button></div></div>}
            {prompt === null && <p className="prompt">You’ve finished the prompts. Search for another title to refine your picks.</p>}
          </div>
          <div className="panel taste search-panel"><h2>Find a film you know</h2>
            <label className="search-label"><span aria-hidden="true">⌕</span><input value={search} type="search" placeholder="Search by title" aria-label="Search by title"
              onChange={event => { setSearch(event.target.value); setSelected(undefined) }} /></label>
            {searchError && <p className="movie-meta" role="status">{searchError}</p>}
            {results.length > 0 && <ul className="search-results">{results.map(movie => <li key={movie.id}>
              <button type="button" onClick={() => { setSelected(movie); setSearch(''); setResults([]) }}>
                <strong>{movie.title}</strong><MovieMeta movie={movie} />
              </button></li>)}</ul>}
            {selected && <div className="selected"><strong>{selected.title}</strong><Stars movie={selected} value={profile.ratings[selected.id]?.stars} onRate={stars => choose(selected, stars)} /></div>}
            <div className="profile-info"><span>{ratingCount} {ratingCount === 1 ? 'rating' : 'ratings'} added</span>
              {(ratingCount > 0 || progress > 0) && <button className="text-button" type="button" onClick={reset}>Start over</button>}</div>
          </div>
        </section>
        <section className="recommendations" aria-labelledby="picks-title">
          <div className="list-header"><div><h2 id="picks-title">{ratingCount ? 'Your shortlist' : 'Popular picks'}</h2>
            <p>{ratingCount ? 'Based on the films you rated.' : 'Rate a film to start personalizing these picks.'}</p></div>
            <div className="filters"><div><label htmlFor="genre-filter">Genre</label><select id="genre-filter" value={genre} onChange={event => setGenre(event.target.value)}><option value="">All genres</option>{genres.map(item => <option key={item}>{item}</option>)}</select></div>
              <div><label htmlFor="decade-filter">Decade</label><select id="decade-filter" value={decade} onChange={event => setDecade(event.target.value)}><option value="">All decades</option>
                {Array.from({ length: 15 }, (_, i) => 2020 - 10 * i).map(year => <option key={year} value={year}>{year}s</option>)}</select></div></div>
          </div>
          <div aria-live="polite" aria-busy={loading}>
            {recommendError ? <p role="alert">{recommendError}</p> : <>
              {!picks.length && <p role="status">{loading ? 'Finding films…' : 'No films match these filters.'}</p>}
              <ul className="pick-grid">{picks.map(({ movie, because, signal }) => <li key={movie.id}>
                <h3>{movie.title}</h3><MovieMeta movie={movie} />
                <p>{!because ? 'Ranked using catalog popularity.' : signal === 'content' ? `Similar story or genres to ${because}.` : `Viewers who liked ${because} also liked this.`}</p>
              </li>)}</ul>
            </>}
          </div>
        </section>
      </>}
    </main>
    <footer className="shell footer"><span>FilmMatch</span><a href="https://www.cs.cmu.edu/~ark/personas/">CMU plot summaries</a><a href="https://grouplens.org/datasets/movielens/">MovieLens ratings</a></footer>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
