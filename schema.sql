create type user_role as enum ('viewer', 'editor', 'admin');

create table users
(
    id           uuid primary key   default gen_random_uuid(),
    email        text      not null unique,
    first_name   text      not null,
    last_name    text      not null,
    display_name text      not null,
    picture      text      not null,
    role         user_role not null default 'viewer',
    created_at   timestamp not null default now()
);

create table tags
(
    id          serial primary key,
    name        text      not null unique,
    description text,
    created_at  timestamp not null default now(),
    updated_at  timestamp not null default now()
);

-- Tables, Documents, Pages
create table folders
(
    id         uuid primary key     default gen_random_uuid(),
    name       text        not null,
    parent_id  uuid        references folders (id) on delete set null,
    drive_id   text unique not null,
    hidden     boolean     not null default false,
    protected  boolean     not null default false,
    created_at timestamp   not null default now(),
    updated_at timestamp   not null default now()
);

create table documents
(
    id                uuid primary key     default gen_random_uuid(),
    filename          text        not null,
    folder_id         uuid        references folders (id) on delete set null,
    drive_id          text unique not null,
    page_count        int         not null,
    hidden            boolean     not null default false,
    protected         boolean     not null default false,

    -- Document-level Full-Text Search
    aggregated_tokens text        not null default '',
    search_vector     tsvector generated always as (to_tsvector('simple', aggregated_tokens)) stored,

    created_at        timestamp   not null default now(),
    updated_at        timestamp   not null default now()
);

create table pages
(
    id                uuid primary key   default gen_random_uuid(),
    document_id       uuid      not null references documents (id) on delete cascade,
    page_number       int       not null,
    content           text      not null,

    -- Page-level Full-Text Search (for highlighting)
    tokenized_content text      not null default '',
    search_vector     tsvector generated always as (to_tsvector('simple', tokenized_content)) stored,

    created_at        timestamp not null default now(),
    updated_at        timestamp not null default now(),
    unique (document_id, page_number)
);

-- Many-to-Many Junction Tables
create table folder_tags
(
    folder_id uuid references folders (id) on delete cascade,
    tag_id    int references tags (id) on delete cascade,
    primary key (folder_id, tag_id)
);

create table document_tags
(
    document_id uuid references documents (id) on delete cascade,
    tag_id      int references tags (id) on delete cascade,
    primary key (document_id, tag_id)
);

create table page_tags
(
    page_id uuid references pages (id) on delete cascade,
    tag_id  int references tags (id) on delete cascade,
    primary key (page_id, tag_id)
);

-- Strict Audit Logs
create table search_logs
(
    id               serial primary key,
    user_id          uuid      not null references users (id) on delete restrict,
    search_query     text      not null,
    search_hidden    boolean   not null,
    search_protected boolean   not null,
    search_at        timestamp not null default now()
);

create table access_logs
(
    id          serial primary key,
    user_id     uuid      not null references users (id) on delete restrict,
    document_id uuid      not null references documents (id) on delete restrict,
    page_number int,
    access_at   timestamp not null default now()
);

create table error_logs
(
    id         serial primary key,
    process    text,
    message    text,
    created_at timestamp not null default now()
);

-- Performance Indexes

-- GIN Indexes for Full-Text Search speed
create index idx_documents_search on documents using gin (search_vector);
create index idx_pages_search on pages using gin (search_vector);

-- B-Tree Indexes for Foreign Keys (crucial for JOIN and WHERE clause performance)
create index idx_folders_parent_id on folders (parent_id);
create index idx_documents_folder_id on documents (folder_id);
create index idx_pages_document_id on pages (document_id);
create index idx_search_logs_user_id on search_logs (user_id);
create index idx_access_logs_user_id on access_logs (user_id);
create index idx_access_logs_document_id on access_logs (document_id);