import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

type Movie = { id: number; title: string; year: number | null; genres: string[]; rating_count: number }
type Pick = { movie: Movie; because: string | null }
type Profile = { ratings: Record<number, { movie: Movie; stars: number }>; answered: number[] }
type Metrics = {
  assigned_users: number; eligible_users: number; popularity_ndcg_at_10?: number;
  itemknn_ndcg_at_10?: number; paired_lift?: number; paired_bootstrap_95_ci?: number[];
  simulated_skip_rate?: number;
}
type Evidence = { dataset: string; sampled_users: number; code_sha: string; seed: number; metrics: Record<string, Metrics> }

const empty: Profile = { ratings: {}, answered: [] }
const storageKey = 'filmmatch.profile.v1'

function initialProfile(): Profile {
  try {
    const data = JSON.parse(localStorage.getItem(storageKey) ?? 'null') as Profile | null
    return data?.ratings && Array.isArray(data.answered) ? data : empty
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
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    api<Evidence>('/evidence', { signal: controller.signal }).then(setReport).catch(reason => {
      if (!controller.signal.aborted) setError(reason.message)
    })
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
      <p className="movie-meta">Source revision <code>{report.code_sha}</code></p>
    </>}
  </section>
}

function App() {
  const [tab, setTab] = useState<'discover' | 'benchmark'>('discover')
  const [profile, setProfile] = useState(initialProfile)
  const [prompts, setPrompts] = useState<Movie[]>()
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
    Promise.all([api<Movie[]>('/prompts', { signal: controller.signal }),
      api<{ genres: string[] }>('/catalog', { signal: controller.signal })]).then(([movies, catalog]) => {
      setPrompts(movies); setGenres(catalog.genres)
    }).catch(reason => { if (!controller.signal.aborted) setInitialError(reason.message) })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(profile)) } catch { /* Session still works in memory. */ }
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

  const prompt = prompts?.find(movie => !profile.answered.includes(movie.id) && !profile.ratings[movie.id])
  const progress = prompts?.filter(movie => profile.answered.includes(movie.id) || profile.ratings[movie.id]).length ?? 0
  const ratingCount = Object.keys(profile.ratings).length
  const choose = (movie: Movie, stars: number) => {
    setProfile(current => ({ ...current, ratings: { ...current.ratings, [movie.id]: { movie, stars } } }))
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
            <div className="panel-top"><h2>Rate a movie</h2>{prompts && <span className="progress-label">{progress} / {prompts.length}</span>}</div>
            {prompts && <div className="progress" aria-hidden="true">{prompts.map((movie, i) => <span key={movie.id} className={i < progress ? 'done' : ''} />)}</div>}
            {!prompts && !initialError && <p role="status">Loading your first prompt…</p>}
            {prompt && <div className="prompt"><h3>{prompt.title}</h3><MovieMeta movie={prompt} />
              <div className="prompt-actions"><Stars movie={prompt} onRate={stars => choose(prompt, stars)} />
                <button type="button" className="text-button" onClick={() => setProfile(current => ({
                  ...current, answered: [...current.answered, prompt.id],
                }))}>Haven’t seen it</button></div></div>}
            {prompts && !prompt && <p className="prompt">You’ve finished the prompts. Search for another title to refine your picks.</p>}
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
            <div className="filters"><label>Genre<select value={genre} onChange={event => setGenre(event.target.value)}><option value="">All genres</option>{genres.map(item => <option key={item}>{item}</option>)}</select></label>
              <label>Decade<select value={decade} onChange={event => setDecade(event.target.value)}><option value="">All decades</option>
                {Array.from({ length: 15 }, (_, i) => 2020 - 10 * i).map(year => <option key={year} value={year}>{year}s</option>)}</select></label></div>
          </div>
          <div aria-live="polite" aria-busy={loading}>
            {recommendError ? <p role="alert">{recommendError}</p> : <>
              {!picks.length && <p role="status">{loading ? 'Finding films…' : 'No films match these filters.'}</p>}
              <ul className="pick-grid">{picks.map(({ movie, because }) => <li key={movie.id}>
                <h3>{movie.title}</h3><MovieMeta movie={movie} />
                <p>{because ? `Viewers who liked ${because} also liked this.` : 'Ranked using catalog popularity.'}</p>
              </li>)}</ul>
            </>}
          </div>
        </section>
      </>}
    </main>
    <footer className="shell footer"><span>FilmMatch</span><a href="https://grouplens.org/datasets/movielens/">Ratings data from MovieLens</a></footer>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
