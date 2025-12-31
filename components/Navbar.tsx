"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Sparkles } from 'lucide-react';
import { useEffect, useState } from 'react';

export default function Navbar() {
    const pathname = usePathname();
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <nav
            className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 border-b ${scrolled
                    ? 'py-4 bg-black/95 backdrop-blur-2xl border-white/10 shadow-2xl'
                    : 'py-6 bg-transparent border-transparent'
                }`}
        >
            <div className="max-w-7xl mx-auto px-6 sm:px-10 lg:px-20 flex items-center justify-between">
                <Link href="/" className="flex items-center gap-2 group">
                    <Sparkles className="w-5 h-5 text-primary group-hover:rotate-12 transition-transform" />
                    <span className="font-bold tracking-tight text-lg">
                        <span className="gradient-text">Scientific</span> Spaces
                    </span>
                </Link>

                <div className="flex items-center gap-8 text-sm font-medium">
                    <Link
                        href="/"
                        className={`transition-colors hover:text-primary ${pathname === '/' ? 'text-primary' : 'text-foreground/70'}`}
                    >
                        Home
                    </Link>
                    <Link
                        href="/blog"
                        className={`transition-colors hover:text-primary ${pathname.startsWith('/blog') ? 'text-primary' : 'text-foreground/70'}`}
                    >
                        Blog
                    </Link>
                </div>
            </div>
        </nav>
    );
}
