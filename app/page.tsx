import Link from 'next/link';
import { ArrowRight, Sparkles, BookOpen, Atom } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden px-6">
      {/* Background Decor */}
      <div className="absolute top-0 left-0 w-full h-full -z-10 bg-black">
        <div className="absolute top-[10%] left-[10%] w-[40%] h-[40%] bg-primary/20 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-[10%] right-[10%] w-[40%] h-[40%] bg-accent/20 rounded-full blur-[120px] animate-pulse delay-1000" />
      </div>

      <main className="max-w-4xl text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8 animate-fade-in">
          <Sparkles className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium">New Research Published</span>
        </div>

        <h1 className="text-6xl sm:text-7xl lg:text-8xl font-bold tracking-tight mb-8 animate-slide-up">
          <span className="gradient-text">Scientific</span> Spaces
        </h1>

        <p className="text-xl text-muted mb-12 max-w-2xl mx-auto leading-relaxed animate-slide-up delay-100">
          A platform dedicated to exploring the intersection of advanced mathematics,
          deep learning optimization, and the future of LLM architecture.
        </p>

        <div className="flex flex-col sm:flex-row gap-6 justify-center animate-slide-up delay-200">
          <Link
            href="/blog"
            className="group bg-primary hover:bg-primary-hover text-white px-10 py-4 rounded-full font-bold transition-all shadow-lg shadow-primary/25 flex items-center gap-2 justify-center"
          >
            Explore Blogs <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>

          <button className="glass px-10 py-4 rounded-full font-bold hover:bg-white/10 transition-colors flex items-center gap-2 justify-center">
            View Research <Atom className="w-5 h-5" />
          </button>
        </div>

        <div className="mt-24 grid grid-cols-1 sm:grid-cols-3 gap-8 animate-fade-in delay-500">
          <div className="glass p-6 rounded-2xl text-left">
            <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center mb-4">
              <BookOpen className="w-6 h-6 text-primary" />
            </div>
            <h3 className="font-bold mb-2">In-depth Analysis</h3>
            <p className="text-sm text-muted">Rigorous derivation and theoretical insights into modern AI.</p>
          </div>
          <div className="glass p-6 rounded-2xl text-left">
            <div className="w-10 h-10 rounded-lg bg-accent/20 flex items-center justify-center mb-4">
              <Sparkles className="w-6 h-6 text-accent" />
            </div>
            <h3 className="font-bold mb-2">Cutting Edge</h3>
            <p className="text-sm text-muted">Translating latest findings from global research communities.</p>
          </div>
          <div className="glass p-6 rounded-2xl text-left">
            <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center mb-4">
              <Atom className="w-6 h-6 text-foreground" />
            </div>
            <h3 className="font-bold mb-2">Math Driven</h3>
            <p className="text-sm text-muted">Proving concepts from first principles of calculus and statistics.</p>
          </div>
        </div>
      </main>
    </div>
  );
}
