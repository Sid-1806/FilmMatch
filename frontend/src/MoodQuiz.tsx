import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import type { Movie } from './main'
import { MovieCardTitle } from './MovieDetails'

type Option = { id: string; label: string }
type QuestionText = { label: string; options: Option[] }
type Question = QuestionText & {
  id: string
  variant?: QuestionText & { when: { question: string; option: string } }
}
type Answers = Record<string, string>
type MoodPick = { movie: Movie; reasons: { answer: string; detail: string }[] }

function questionForAnswers(question: Question, answers: Answers) {
  const variant = question.variant
  return variant && answers[variant.when.question] === variant.when.option
    ? { ...question, label: variant.label, options: variant.options }
    : question
}

export default function MoodQuiz({ active, onOpenMovie }: { active: boolean; onOpenMovie: (movie: Movie) => void }) {
  const [questions, setQuestions] = useState<Question[]>()
  const [questionError, setQuestionError] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const [answers, setAnswers] = useState<Answers>({})
  const [step, setStep] = useState(0)
  const [editing, setEditing] = useState(false)
  const [picks, setPicks] = useState<MoodPick[]>()
  const [pending, setPending] = useState(false)
  const [resultError, setResultError] = useState(false)
  const stepHeading = useRef<HTMLHeadingElement>(null)

  useEffect(() => {
    if (!active || questions) return
    const controller = new AbortController()
    api<Question[]>('/mood/questions', { signal: controller.signal }).then(setQuestions)
      .catch(() => { if (!controller.signal.aborted) setQuestionError(true) })
    return () => controller.abort()
  }, [active, questions, attempt])

  useEffect(() => {
    if (active) stepHeading.current?.focus()
  }, [active, step, picks, questions, editing])

  const visibleQuestions = questions?.map(question => questionForAnswers(question, answers))
  const question = visibleQuestions?.[step]

  async function recommend(updated: Answers) {
    setPending(true)
    setResultError(false)
    try {
      const result = await api<MoodPick[]>('/mood/recommend', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers: updated }),
      })
      setAnswers(updated)
      setEditing(false)
      setPicks(result)
    } catch {
      setResultError(true)
    } finally {
      setPending(false)
    }
  }

  function choose(option: string) {
    if (!question || !questions || pending) return
    const updated = { ...answers, [question.id]: option }
    for (const other of questions) {
      if (other.variant?.when.question !== question.id || !updated[other.id]) continue
      const activeOptions = questionForAnswers(other, updated).options
      if (!activeOptions.some(candidate => candidate.id === updated[other.id])) updated[other.id] = 'any'
    }
    if (editing || step === questions.length - 1) void recommend(updated)
    else {
      setAnswers(updated)
      setStep(step + 1)
    }
  }

  function startOver() {
    setAnswers({})
    setStep(0)
    setEditing(false)
    setPicks(undefined)
    setResultError(false)
  }

  const showingResults = picks !== undefined && !editing

  return <section className="mood" aria-labelledby="mood-title" hidden={!active}>
    <div className="mood-intro">
      <h1 id="mood-title">What are you in the <em>mood for?</em></h1>
      <p className="lead">Tell us what sounds good tonight. FilmMatch will match your answers to genres and plot summaries.</p>
    </div>
    {questionError && !questions && <p role="alert">Couldn’t load the questions. <button className="text-button" type="button"
      onClick={() => { setQuestionError(false); setAttempt(attempt + 1) }}>Try again</button></p>}
    {!questions && !questionError && <p role="status">Loading questions…</p>}
    {questions && question && !showingResults && <div className="panel mood-card">
      <div className="mood-progress">
        <p>Question {step + 1} of {questions.length}</p>
        <progress max={questions.length} value={Object.keys(answers).length} aria-label="Questions answered" />
      </div>
      <h2 id="mood-question" tabIndex={-1} ref={stepHeading}>{question.label}</h2>
      <div className="mood-options" role="group" aria-labelledby="mood-question" aria-busy={pending}>
        {question.options.map(option => <button className="mood-option" key={option.id} type="button" disabled={pending}
          aria-pressed={answers[question.id] === option.id} onClick={() => choose(option.id)}>
          {option.label}<span aria-hidden="true">›</span>
        </button>)}
      </div>
      <div className="mood-actions">
        {editing ? <button className="text-button" type="button" disabled={pending} onClick={() => {
          setEditing(false); setResultError(false)
        }}>Back to results</button> :
          <button className="text-button" type="button" disabled={pending || step === 0} onClick={() => {
            setStep(step - 1); setResultError(false)
          }}>Back</button>}
        {pending && <p role="status">Finding your films…</p>}
      </div>
      {resultError && <p role="alert">Couldn’t find your picks right now. Choose an option to try again.</p>}
    </div>}
    {visibleQuestions && showingResults && <section className="mood-results" aria-labelledby="mood-results-title">
      <h2 id="mood-results-title" tabIndex={-1} ref={stepHeading}>Picks for tonight</h2>
      <p className="movie-meta">{picks.length
        ? `${picks.length} ${picks.length === 1 ? 'film' : 'films'} for this mood`
        : 'No films match these filters. Edit an answer to widen the search.'}</p>
      <details className="panel mood-review">
        <summary>Edit answers</summary>
        <ul>{visibleQuestions.map((item, index) => <li key={item.id}>
          <span id={`mood-review-${item.id}`}>{item.label}</span>
          <button className="text-button" type="button" aria-describedby={`mood-review-${item.id}`}
            onClick={() => { setStep(index); setEditing(true) }}>
            {item.options.find(option => option.id === answers[item.id])?.label}{' '}
            <span className="mood-edit-label">Edit</span>
          </button>
        </li>)}</ul>
      </details>
      <ul className="pick-grid">{picks.map(({ movie, reasons }) => <li className="movie-card" key={movie.id}>
        <MovieCardTitle movie={movie} onOpen={onOpenMovie} />
        <span className="movie-meta">{[movie.year, ...movie.genres.slice(0, 3)].filter(Boolean).join(' · ')}</span>
        {reasons.map(({ answer, detail }) => <p key={answer}><strong>{answer}</strong><br />{detail}</p>)}
      </li>)}</ul>
      <button type="button" className="text-button mood-restart" onClick={startOver}>Start a new quiz</button>
    </section>}
  </section>
}
