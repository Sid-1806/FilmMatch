import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import type { Movie } from './main'

type Details = {
  overview: string | null
  release_date: string | null
  runtime_minutes: number | null
  directors: string[]
  cast: string[]
  poster_url: string | null
  imdb_rating: number | null
  imdb_url: string | null
  tmdb_rating: number | null
}

type MovieOpener = { movie: Movie; onOpen: (movie: Movie) => void }

export function MovieCardTitle({ movie, onOpen }: MovieOpener) {
  return <h3><button className="movie-card-action" type="button" aria-haspopup="dialog"
    onClick={() => onOpen(movie)}>{movie.title}</button></h3>
}

export default function MovieDetails({ movie, onClose }: { movie: Movie; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [details, setDetails] = useState<Details>()
  const [error, setError] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const [posterFailed, setPosterFailed] = useState(false)
  const poster = details?.poster_url && !posterFailed ? details.poster_url : null

  useEffect(() => {
    const element = dialog.current
    element?.showModal()
    return () => element?.close()
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    api<Details>(`/movies/${movie.id}`, { signal: controller.signal }).then(setDetails)
      .catch(() => { if (!controller.signal.aborted) setError(true) })
    return () => controller.abort()
  }, [attempt, movie.id])

  function close() {
    dialog.current?.close()
    onClose()
  }

  const released = details?.release_date
    ? new Date(`${details.release_date}T00:00:00Z`).toLocaleDateString(undefined, { dateStyle: 'medium', timeZone: 'UTC' })
    : null

  return <dialog ref={dialog} className="movie-dialog" aria-labelledby="movie-dialog-title"
    onCancel={event => { event.preventDefault(); close() }}
    onClick={event => { if (event.target === event.currentTarget) close() }}>
    <article className="movie-dialog-panel">
      <div className="movie-dialog-heading">
        <div><h2 id="movie-dialog-title">{movie.title}</h2>
          <p className="movie-meta">{[movie.year, ...movie.genres].filter(Boolean).join(' · ')}</p></div>
        <button className="movie-dialog-close" type="button" aria-label="Close movie details" onClick={close}>×</button>
      </div>
      {!details && !error && <p role="status">Loading movie details…</p>}
      {error && <p role="alert">Couldn’t load movie details. <button className="text-button" type="button"
        onClick={() => { setError(false); setAttempt(attempt + 1) }}>Try again</button></p>}
      {details && <div className={poster ? 'movie-dialog-body with-poster' : 'movie-dialog-body'}>
        {poster && <img className="movie-dialog-poster" src={poster} alt={`Poster for ${movie.title}`}
          onError={() => setPosterFailed(true)} width="342" height="513" />}
        <div className="movie-dialog-copy">
          <p className="movie-overview">{details.overview || 'No plot summary available.'}</p>
          <dl className="movie-facts">
            <div><dt>IMDb</dt><dd>{details.imdb_rating ? <strong>{details.imdb_rating.toFixed(1)}/10</strong> : 'Unavailable'}</dd></div>
            {details.tmdb_rating !== null && <div><dt>TMDB</dt><dd>{details.tmdb_rating.toFixed(1)}/10</dd></div>}
            {details.runtime_minutes && <div><dt>Runtime</dt><dd>{details.runtime_minutes} min</dd></div>}
            {released && <div><dt>Release date</dt><dd>{released}</dd></div>}
            {details.directors.length > 0 && <div><dt>Director</dt><dd>{details.directors.join(', ')}</dd></div>}
            {details.cast.length > 0 && <div><dt>Cast</dt><dd>{details.cast.join(', ')}</dd></div>}
          </dl>
          <div className="movie-detail-links">
            {details.imdb_url && <a href={details.imdb_url} target="_blank" rel="noreferrer">View on IMDb</a>}
            {details.imdb_rating !== null && <a href="https://www.omdbapi.com/" target="_blank" rel="noreferrer">IMDb score via OMDb</a>}
          </div>
        </div>
      </div>}
    </article>
  </dialog>
}
