-- HITL (Human-In-The-Loop) Reviews Table
-- Stores queries that require human review due to low confidence or sensitive content

CREATE TABLE IF NOT EXISTS hitl_reviews (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    retrieved_docs JSONB,
    similarity_score FLOAT,
    reason TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP,
    reviewer TEXT,
    final_answer TEXT,
    lang TEXT DEFAULT 'en' CHECK (lang IN ('en', 'zhtw')),
    character TEXT DEFAULT 'hr' CHECK (character IN ('hr', 'engineer'))
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_hitl_status ON hitl_reviews(status);
CREATE INDEX IF NOT EXISTS idx_hitl_created_at ON hitl_reviews(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_hitl_session_id ON hitl_reviews(session_id);

-- Comments for documentation
COMMENT ON TABLE hitl_reviews IS 'Audit trail for queries requiring human review';
COMMENT ON COLUMN hitl_reviews.id IS 'Unique identifier for the review request';
COMMENT ON COLUMN hitl_reviews.session_id IS 'User session ID from the chat system';
COMMENT ON COLUMN hitl_reviews.question IS 'Original user query that triggered HITL';
COMMENT ON COLUMN hitl_reviews.retrieved_docs IS 'JSON array of retrieved documents with scores';
COMMENT ON COLUMN hitl_reviews.similarity_score IS 'Average similarity score from vector retrieval';
COMMENT ON COLUMN hitl_reviews.reason IS 'Why HITL was triggered (low confidence, sensitive content, etc.)';
COMMENT ON COLUMN hitl_reviews.status IS 'Current status: pending, approved, or rejected';
COMMENT ON COLUMN hitl_reviews.created_at IS 'When the review was requested';
COMMENT ON COLUMN hitl_reviews.reviewed_at IS 'When the review was completed';
COMMENT ON COLUMN hitl_reviews.reviewer IS 'Username or ID of the reviewer';
COMMENT ON COLUMN hitl_reviews.final_answer IS 'Approved answer provided by human reviewer';
COMMENT ON COLUMN hitl_reviews.lang IS 'Language of the query (en or zhtw)';
COMMENT ON COLUMN hitl_reviews.character IS 'Interviewer persona (hr or engineer)';
