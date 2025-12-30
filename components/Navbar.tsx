"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Sparkles } from 'lucide-react';

export default function Navbar() {
    const pathname = usePathname();

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 flex justify-center p-6 pointer-events-none">
            <div className="glass px-8 py-3 rounded-full flex items-center gap-8 pointer-events-auto shadow-2xl shadow-primary/10">
                <Link href="/" className="flex items-center gap-2 group">
                    <Sparkles className="w-5 h-5 text-primary group-hover:rotate-12 transition-transform" />
                    <span className="font-bold tracking-tight text-lg">
                        <span className="gradient-text">Scientific</span> Spaces
                    </span>
                </Link>

                <div className="h-4 w-[1px] bg-white/10" />

                <div className="flex items-center gap-6">
                    <Link
                        href="/"
                        className={`text-sm font-medium transition-colors hover:text-primary ${pathname === '/' ? 'text-primary font-bold' : 'text-foreground/70'}`}
                    >
                        Home
                    </Link>
                    <Link
                        href="/blog"
                        className={`text-sm font-medium transition-colors hover:text-primary ${pathname.startsWith('/blog') ? 'text-primary font-bold' : 'text-foreground/70'}`}
                    >
                        Blog
                    </Link>
                </div>
            </div>
        </nav>
    );
}
