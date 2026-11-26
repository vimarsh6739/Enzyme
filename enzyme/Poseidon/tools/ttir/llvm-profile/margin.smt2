; Model the exact real arithmetic of the TTIR margin and Enzyme's product rule.
; The Herbie frontier includes 1 - (-x - x) for the value and dx + dx for its
; JVP. Both identities hold for all real inputs. This theorem omits FP rounding.
(set-logic QF_NRA)
(declare-const x Real)
(declare-const dx Real)
(define-fun r () Real (+ x 1))
(define-fun value () Real (- (* r r) (* x x)))
(define-fun jvp () Real (- (+ (* dx r) (* r dx)) (+ (* dx x) (* x dx))))
(assert (or (distinct value (- 1 (- (- x) x))) (distinct jvp (+ dx dx))))
(check-sat)
