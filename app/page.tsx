import Link from 'next/link';
import { Sparkles, Calendar, User, ArrowRight, Info } from 'lucide-react';
import { getSortedPostsData } from '@/lib/blog';
import { formatDate } from 'date-fns';

export default function Home() {
  const allPostsData = getSortedPostsData();

  return (
    <div className="min-h-screen flex flex-col items-center relative overflow-hidden px-6 pt-20 pb-20">
      {/* Background Decor */}
      <div className="absolute top-0 left-0 w-full h-full -z-10 bg-black">
        <div className="absolute top-[-10%] left-[-10%] w-[60%] h-[60%] bg-primary/10 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute top-[20%] right-[-10%] w-[50%] h-[50%] bg-accent/10 rounded-full blur-[120px] animate-pulse delay-1000" />
      </div>

      <header className="max-w-4xl text-center mb-24 mt-20">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8 animate-fade-in">
          <Sparkles className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium">Independent Research Archive</span>
        </div>

        <h1 className="text-6xl sm:text-7xl lg:text-8xl font-bold tracking-tight mb-8 animate-slide-up">
          <span className="gradient-text">Scientific</span> Spaces
        </h1>

        <p className="text-xl text-muted max-w-2xl mx-auto leading-relaxed animate-slide-up delay-100">
          Exploring the intersection of advanced mathematics,
          deep learning optimization, and the future of LLM architecture.
        </p>
      </header>

      <main className="w-full max-w-7xl mx-auto animate-fade-in delay-300">
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
                  <span className="text-[10px] uppercase tracking-wider bg-green-500/10 text-green-500 px-2 py-1 rounded-md font-bold border border-green-500/20 flex items-center gap-1">
                    <span className="w-1 h-1 bg-green-500 rounded-full" />
                    100% Human
                  </span>
                </div>
              </article>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
