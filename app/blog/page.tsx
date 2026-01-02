import Link from 'next/link';
import { getSortedPostsData } from '@/lib/blog';
import { formatDate } from 'date-fns';
import { Calendar, User, ArrowRight, Info } from 'lucide-react';

export default function BlogIndex() {
    const allPostsData = getSortedPostsData();

    return (
        <div className="min-h-screen py-20 px-6 sm:px-10 lg:px-20 max-w-7xl mx-auto">
            <header className="mb-16">
                <h1 className="text-5xl font-bold mb-4 tracking-tight">
                    <span className="gradient-text">Scientific</span> Spaces
                </h1>
                <p className="text-muted text-lg max-w-2xl">
                    Exploring the frontiers of mathematics, optimization, and large language models.
                </p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {allPostsData.map((post) => (
                    <Link
                        key={post.slug}
                        href={`/blog/${post.slug}`}
                        className="group block"
                    >
                        <article className="glass h-full rounded-2xl p-8 flex flex-col">
                            <div className="flex items-center gap-2 text-primary text-sm font-medium mb-4">
                                <Calendar className="w-4 h-4" />
                                {formatDate(new Date(post.date), 'MMMM d, yyyy')}
                            </div>

                            <h2 className="text-2xl font-bold mb-4 group-hover:text-primary transition-colors line-clamp-2">
                                {post.title}
                            </h2>

                            <p className="text-foreground/70 mb-8 line-clamp-3 text-sm leading-relaxed">
                                {post.excerpt}
                            </p>

                            {post.credit && (
                                <div className="mb-6 flex items-start gap-2 text-[11px] text-muted italic">
                                    <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
                                    <span className="line-clamp-1">{post.credit}</span>
                                </div>
                            )}

                            <div className="mt-auto pt-6 border-t border-white/5 flex items-center justify-between">
                                <div className="flex items-center gap-2 text-foreground/60 text-sm">
                                    <User className="w-4 h-4" />
                                    {post.author}
                                </div>

                                <div className="flex items-center gap-1 text-primary text-sm font-semibold opacity-0 group-hover:opacity-100 transition-all transform translate-x-2 group-hover:translate-x-0">
                                    Read More <ArrowRight className="w-4 h-4" />
                                </div>
                            </div>

                            <div className="flex flex-wrap gap-2 mt-4 items-center">
                                {post.tags.slice(0, 3).map(tag => (
                                    <span key={tag} className="text-[10px] uppercase tracking-wider bg-primary/10 text-primary/80 px-2 py-1 rounded-md font-bold">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </article>
                    </Link>
                ))}
            </div>
        </div>
    );
}
