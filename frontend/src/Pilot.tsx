import { useEffect, useState } from 'react'
import { api } from './api'
import type { Movie } from './main'

type Event = { action: 'rate' | 'pick' | 'next' | 'prefer'; movie_id?: number; stars?: number; preference?: 'first' | 'second' | 'tie' }
type Session = { revision: number; phase: 'prompt' | 'shortlist' | 'between' | 'preference' | 'complete';
  round: number; answered: number; prompt_limit: number; prompt?: Movie; picks?: Movie[] }

const storageKey = 'filmmatch.pilot.five-prompts.v1'

function savedToken() {
  try { return localStorage.getItem(storageKey) ?? '' } catch { return '' }
}

function pilotRequest(path: string, token: string, body?: object) {
  return api<Session>(`/pilot/${path}`, {
    method: body ? 'POST' : 'GET', headers: { 'X-Pilot-Session': token, 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
}

export default function Pilot() {
  const [token, setToken] = useState(savedToken)
  const [session, setSession] = useState<Session>()
  const [pending, setPending] = useState(Boolean(token))
  const [error, setError] = useState('')

  useEffect(() => {
    const existing = savedToken()
    if (existing) pilotRequest('session', existing).then(setSession).catch(() => undefined).finally(() => setPending(false))
  }, [])

  async function start() {
    const key = token || crypto.randomUUID()
    setToken(key)
    try { localStorage.setItem(storageKey, key) } catch { /* The current page can still complete the pilot. */ }
    setPending(true); setError('')
    try { setSession(await pilotRequest('start', key, { consent: true })) }
    catch { setError('Could not start the pilot. Try again.') }
    finally { setPending(false) }
  }

  async function record(event: Event) {
    if (!session || pending) return
    setPending(true); setError('')
    try { setSession(await pilotRequest('event', token, { revision: session.revision, ...event })) }
    catch { setError('Could not save this step. Try again.') }
    finally { setPending(false) }
  }

  const prompt = session?.prompt
  return <div className="site">
    <header className="header"><div className="shell topbar"><a className="brand pilot-home" href="/"><span className="brand-mark" aria-hidden="true">▶</span> FilmMatch</a></div></header>
    <main className="shell pilot" aria-busy={pending}>
      <h1>Find a film.<br /><em>Try two ways.</em></h1>
      {!session && <section className="panel pilot-panel">
        <h2>FilmMatch pilot</h2>
        <p>Try two rounds of five movie prompts. After each round, choose one film you would watch or say that none fit.</p>
        <p>This optional study records anonymous ratings, skips, choices, and timing. Combined results may appear in the FilmMatch case study.</p>
        <button type="button" className="lab-run" onClick={start} disabled={pending}>Agree and start</button>
      </section>}
      {session && ['prompt', 'shortlist'].includes(session.phase) && <p className="pilot-progress">Round {session.round} of 2 · {session.answered} / {session.prompt_limit} prompts answered</p>}
      {session?.phase === 'prompt' && prompt && <section className="panel pilot-panel">
        <h2>{prompt.title}</h2>
        <p className="movie-meta">{[prompt.year, ...prompt.genres.slice(0, 2)].filter(Boolean).join(' · ')}</p>
        <div className="prompt-actions"><div className="stars" role="group" aria-label={`Rate ${prompt.title}`}>
          {[1, 2, 3, 4, 5].map(stars => <button key={stars} type="button" aria-label={`${stars} ${stars === 1 ? 'star' : 'stars'}`} disabled={pending}
            onClick={() => record({ action: 'rate', movie_id: prompt.id, stars })}>★</button>)}
        </div><button type="button" className="text-button" disabled={pending} onClick={() => record({ action: 'rate', movie_id: prompt.id })}>Haven’t seen it</button></div>
      </section>}
      {session?.phase === 'shortlist' && <section className="pilot-shortlist">
        <h2>Which film would you watch?</h2>
        <ul className="pick-grid">{session.picks?.map(movie => <li key={movie.id}>
          <button className="pilot-pick" type="button" disabled={pending} onClick={() => record({ action: 'pick', movie_id: movie.id })}>
            <strong>{movie.title}</strong><span className="movie-meta">{[movie.year, ...movie.genres.slice(0, 2)].filter(Boolean).join(' · ')}</span>
          </button>
        </li>)}</ul>
        <button type="button" className="text-button" disabled={pending} onClick={() => record({ action: 'pick' })}>None of these fit</button>
      </section>}
      {session?.phase === 'between' && <section className="panel pilot-panel">
        <h2>First round complete</h2><p>The second round starts with a fresh taste profile.</p>
        <button type="button" className="lab-run" disabled={pending} onClick={() => record({ action: 'next' })}>Start round two</button>
      </section>}
      {session?.phase === 'preference' && <section className="panel pilot-panel">
        <h2>Which round worked better for you?</h2>
        <div className="pilot-actions">{([['first', 'First round'], ['second', 'Second round'], ['tie', 'No preference']] as const).map(([preference, label]) =>
          <button className="lab-run" key={preference} type="button" disabled={pending} onClick={() => record({ action: 'prefer', preference })}>{label}</button>)}</div>
      </section>}
      {session?.phase === 'complete' && <section className="panel pilot-panel">
        <h2>Thanks for taking part</h2><p>Your pilot responses are saved.</p><a className="text-button" href="/">Open FilmMatch</a>
      </section>}
      {error && <p className="pilot-progress" role="alert">{error}</p>}
    </main>
  </div>
}
