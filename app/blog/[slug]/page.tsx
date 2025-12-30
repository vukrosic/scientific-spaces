import { getPostData, getAllPostSlugs } from '@/lib/blog';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { formatDate } from 'date-fns';
import Link from 'next/link';
import { Calendar, User, ChevronLeft, Share2, Info } from 'lucide-react';

export async function generateStaticParams() {
    const posts = getAllPostSlugs();
    return posts.map((post) => ({
        slug: post.params.slug,
    }));
}

export default async function BlogPost({ params }: { params: Promise<{ slug: string }> }) {
    const { slug } = await params;
    const post = getPostData(slug);

    return (
        <article className="min-h-screen py-20 px-6 sm:px-10 lg:px-20 max-w-4xl mx-auto">
            <Link
                href="/blog"
                className="inline-flex items-center gap-2 text-muted hover:text-primary transition-colors mb-12 group"
            >
                <ChevronLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
                Back to Blog
            </Link>

            <header className="mb-16">
                <div className="flex flex-wrap gap-4 items-center text-muted text-sm mb-6">
                    <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4" />
                        {formatDate(new Date(post.date), 'MMMM d, yyyy')}
                    </div>
                    <div className="w-1 h-1 bg-muted/40 rounded-full" />
                    <div className="flex items-center gap-2">
                        <User className="w-4 h-4" />
                        {post.author}
                    </div>
                    <div className="w-1 h-1 bg-muted/40 rounded-full" />
                    <div className="flex gap-2">
                        {post.tags.map(tag => (
                            <span key={tag} className="text-[10px] uppercase tracking-wider text-primary/80 font-bold">
                                #{tag}
                            </span>
                        ))}
                    </div>
                </div>

                <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-8 leading-tight">
                    {post.title}
                </h1>

                <div className="flex items-center justify-between py-6 border-y border-white/10">
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center font-bold text-white shadow-lg shadow-primary/20">
                            {post.author[0]}
                        </div>
                        <div>
                            <div className="font-semibold">{post.author}</div>
                            <div className="text-xs text-muted">Author & Researcher</div>
                        </div>
                    </div>
                    <button className="p-3 rounded-full glass hover:text-primary transition-colors">
                        <Share2 className="w-5 h-5" />
                    </button>
                </div>

                {post.credit && (
                    <div className="mt-8 p-4 rounded-xl bg-primary/5 border border-primary/10 flex gap-3 items-start">
                        <Info className="w-5 h-5 text-primary mt-0.5" />
                        <p className="text-sm text-foreground/80 italic">
                            {post.credit}
                        </p>
                    </div>
                )}
            </header>

            <div className="markdown-content prose prose-invert prose-lg max-w-none">
                <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                >
                    {post.content}
                </ReactMarkdown>
            </div>

            <footer className="mt-20 pt-10 border-t border-white/10">
                <div className="glass rounded-2xl p-8 text-center">
                    <h3 className="text-xl font-bold mb-2">Enjoyed this post?</h3>
                    <p className="text-muted mb-6">Join our community of independent scientists and researchers.</p>
                    <button className="bg-primary hover:bg-primary-hover text-white px-8 py-3 rounded-full font-bold transition-all shadow-lg shadow-primary/25">
                        Subscribe to Newsletter
                    </button>
                </div>
            </footer>
        </article>
    );
}
