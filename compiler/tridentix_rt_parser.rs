//! Linear token cursor: tokens are moved out of the buffer once.

use crate::lexer::Token;

pub struct TokenCursor {
    tokens: Vec<Token>,
    pos: usize,
}

impl TokenCursor {
    pub fn new(tokens: Vec<Token>) -> Self {
        TokenCursor { tokens, pos: 0 }
    }

    pub fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    /// Drop the current token without cloning its payload.
    pub fn bump(&mut self) {
        if self.pos < self.tokens.len() {
            let _ = std::mem::replace(&mut self.tokens[self.pos], Token::Eof);
            self.pos += 1;
        }
    }

    /// Take ownership of the current token (moves Ident/Str strings once).
    pub fn take(&mut self) -> Token {
        if self.pos >= self.tokens.len() {
            return Token::Eof;
        }
        let tok = std::mem::replace(&mut self.tokens[self.pos], Token::Eof);
        self.pos += 1;
        tok
    }

    pub fn skip_newlines(&mut self) {
        while matches!(self.peek(), Token::Newline) {
            self.bump();
        }
    }

    pub fn eat(&mut self, expected: &Token) -> bool {
        if self.peek() == expected {
            self.bump();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_is_linear_no_double_clone() {
        let mut cur = TokenCursor::new(vec![
            Token::Ident("a".into()),
            Token::Ident("b".into()),
            Token::Eof,
        ]);
        assert!(matches!(cur.peek(), Token::Ident(s) if s == "a"));
        assert_eq!(cur.take(), Token::Ident("a".into()));
        cur.bump();
        assert!(matches!(cur.peek(), Token::Eof));
    }
}
